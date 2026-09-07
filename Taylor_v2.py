#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
LOTO 7/39 — TAYLOROV DISTRIBUCIJSKI REGRESIONI SISTEM

Jedna konačna arhitektura:

1. vremenski zaglađena kontinuirana distribucija svakog broja;
2. Taylorove razlike prvog i drugog reda;
3. HistGradientBoostingRegressor;
4. hronološki izbor parametara;
5. potpuno zamrznuti holdout;
6. blok-bootstrap interval od 95%;
7. Monte Karlo test prema slučajnom očekivanju.

Prvi CSV red predstavlja najstarije izvlačenje.
Poslednji CSV red predstavlja najnovije izvlačenje.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error


# =============================================================================
# PODEŠAVANJA
# =============================================================================

SEED = 39

ZAJEDNICKI_CSV = Path(
    "/data/loto7_4680_k71.csv"
)

BROJ_KUGLICA = 39
BROJ_IZVUCENIH = 7

TEORIJSKA_STOPA = BROJ_IZVUCENIH / BROJ_KUGLICA
SLUCAJNO_OCEKIVANJE = BROJ_IZVUCENIH**2 / BROJ_KUGLICA
UKUPNO_KOMBINACIJA = math.comb(BROJ_KUGLICA, BROJ_IZVUCENIH)

MINIMALNA_ISTORIJA = 250

POLUZIVOTI = (
    10.0,
    20.0,
    40.0,
    80.0,
    160.0,
)

MAKSIMALNA_DUZINA_EWMA = 800

HORIZONT_METE = 5

TEZINE_METE = np.array(
    [1.00, 0.75, 0.55, 0.40, 0.30],
    dtype=float,
)

TEZINE_METE /= TEZINE_METE.sum()

BROJ_VALIDACIONIH_IZVLACENJA = 160
BROJ_HOLDOUT_IZVLACENJA = 200

MAX_TRENING_TRENUTAKA_TUNING = 800
MAX_TRENING_TRENUTAKA_HOLDOUT = 1_000
MAX_TRENING_TRENUTAKA_NEXT = 1_200

BROJ_BOOTSTRAP_PONAVLJANJA = 10_000
BROJ_MONTE_KARLO_PONAVLJANJA = 100_000
BOOTSTRAP_BLOK = 12
NIVO_ZNACAJNOSTI = 0.05

warnings.filterwarnings("ignore")


KONFIGURACIJE_MODELA = (
    {
        "learning_rate": 0.025,
        "max_iter": 400,
        "max_leaf_nodes": 15,
        "min_samples_leaf": 40,
        "l2_regularization": 2.0,
    },
    {
        "learning_rate": 0.035,
        "max_iter": 350,
        "max_leaf_nodes": 31,
        "min_samples_leaf": 50,
        "l2_regularization": 3.0,
    },
    {
        "learning_rate": 0.050,
        "max_iter": 250,
        "max_leaf_nodes": 15,
        "min_samples_leaf": 60,
        "l2_regularization": 4.0,
    },
)


# =============================================================================
# REZULTAT
# =============================================================================

@dataclass
class Rezultat:
    naziv: str
    broj_redova: int
    next_kombinacija: list[int]
    next_skorovi: np.ndarray
    izabrana_konfiguracija: int
    validacioni_prosek: float
    validacioni_mae: float
    holdout_prosek: float
    holdout_medijana: float
    holdout_maksimum: int
    holdout_mae: float
    donja_granica_95: float
    gornja_granica_95: float
    p_vrednost: float
    statisticki_pouzdano: bool
    broj_validacionih_izvlacenja: int
    broj_holdout_izvlacenja: int
    broj_trening_uzoraka: int


# =============================================================================
# UČITAVANJE PODATAKA
# =============================================================================

def ucitaj_csv(
    putanja: Path,
) -> np.ndarray:
    if not putanja.exists():
        raise FileNotFoundError(
            f"CSV fajl ne postoji: {putanja}"
        )

    okvir = pd.read_csv(
        putanja,
        header=None,
    )

    okvir = okvir.apply(
        pd.to_numeric,
        errors="coerce",
    )

    okvir = okvir.dropna(
        axis=0,
        how="all",
    )

    if okvir.shape[1] < BROJ_IZVUCENIH:
        raise ValueError(
            "CSV mora imati najmanje sedam kolona."
        )

    okvir = okvir.iloc[:, :BROJ_IZVUCENIH]

    if okvir.isna().any().any():
        raise ValueError(
            "CSV sadrži vrednosti koje nisu brojevi."
        )

    podaci = okvir.to_numpy(
        dtype=int
    )

    potreban_broj = (
        MINIMALNA_ISTORIJA
        + BROJ_VALIDACIONIH_IZVLACENJA
        + BROJ_HOLDOUT_IZVLACENJA
        + HORIZONT_METE
    )

    if len(podaci) < potreban_broj:
        raise ValueError(
            f"Potrebno je najmanje {potreban_broj} izvlačenja."
        )

    for redni_broj, red in enumerate(
        podaci,
        start=1,
    ):
        if len(np.unique(red)) != BROJ_IZVUCENIH:
            raise ValueError(
                f"Red {redni_broj} nema sedam različitih brojeva."
            )

        if np.any(red < 1) or np.any(
            red > BROJ_KUGLICA
        ):
            raise ValueError(
                f"Red {redni_broj} sadrži broj izvan opsega 1–39."
            )

    return podaci


def napravi_binarnu_matricu(
    izvlacenja: np.ndarray,
) -> np.ndarray:
    binarna = np.zeros(
        (
            len(izvlacenja),
            BROJ_KUGLICA,
        ),
        dtype=np.float64,
    )

    for t, red in enumerate(izvlacenja):
        binarna[t, red - 1] = 1.0

    return binarna


def ravnomerni_trenuci(
    pocetak: int,
    kraj: int,
    maksimum: int,
) -> np.ndarray:
    if kraj <= pocetak:
        raise ValueError(
            "Neispravan hronološki opseg."
        )

    broj = min(
        maksimum,
        kraj - pocetak,
    )

    return np.unique(
        np.linspace(
            pocetak,
            kraj - 1,
            num=broj,
            dtype=int,
        )
    )


# =============================================================================
# ZAGLAĐENA DISTRIBUCIJA
# =============================================================================

def ewma_distribucija(
    binarna: np.ndarray,
    t: int,
    poluzivot: float,
) -> np.ndarray:
    """
    Računa zaglađenu distribuciju koristeći samo redove pre t.
    """

    pocetak = max(
        0,
        t - MAKSIMALNA_DUZINA_EWMA,
    )

    istorija = binarna[
        pocetak:t
    ]

    if len(istorija) == 0:
        return np.full(
            BROJ_KUGLICA,
            TEORIJSKA_STOPA,
            dtype=float,
        )

    starost = np.arange(
        len(istorija) - 1,
        -1,
        -1,
        dtype=float,
    )

    tezine = np.power(
        0.5,
        starost / poluzivot,
    )

    zbir_tezina = float(
        tezine.sum()
    )

    ponderisana_stopa = (
        tezine @ istorija
    ) / zbir_tezina

    # Prior je centriran na teorijskoj stopi 7/39.
    prior_snaga = zbir_tezina

    zagladjena = (
        ponderisana_stopa * zbir_tezina
        + TEORIJSKA_STOPA * prior_snaga
    ) / (
        zbir_tezina + prior_snaga
    )

    return np.asarray(
        zagladjena,
        dtype=float,
    )


# =============================================================================
# TAYLOROVE OSOBINE
# =============================================================================

def taylor_za_poluzivot(
    binarna: np.ndarray,
    t: int,
    poluzivot: float,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """
    Diskretna Taylorova aproksimacija drugog reda:

    T(t+1) = s(t) + Δs(t) + 1/2 Δ²s(t)
    """

    s_t = ewma_distribucija(
        binarna,
        t,
        poluzivot,
    )

    s_t1 = ewma_distribucija(
        binarna,
        t - 1,
        poluzivot,
    )

    s_t2 = ewma_distribucija(
        binarna,
        t - 2,
        poluzivot,
    )

    prva_razlika = (
        s_t - s_t1
    )

    druga_razlika = (
        s_t
        - 2.0 * s_t1
        + s_t2
    )

    projekcija = (
        s_t
        + prva_razlika
        + 0.5 * druga_razlika
    )

    projekcija = np.clip(
        projekcija,
        0.0,
        1.0,
    )

    return (
        s_t,
        prva_razlika,
        druga_razlika,
        projekcija,
    )


def osobine_za_trenutak(
    binarna: np.ndarray,
    t: int,
) -> np.ndarray:
    """
    Pravi po jedan red osobina za svaki od 39 brojeva.
    Nijedna osobina ne koristi izvlačenje t niti buduće redove.
    """

    kolone = []

    for poluzivot in POLUZIVOTI:
        (
            distribucija,
            prva_razlika,
            druga_razlika,
            projekcija,
        ) = taylor_za_poluzivot(
            binarna,
            t,
            poluzivot,
        )

        kolone.extend(
            [
                distribucija,
                prva_razlika,
                druga_razlika,
                projekcija,
            ]
        )

    # Saglasnost Taylorovih projekcija kroz više skala.
    projekcije = np.vstack(
        [
            taylor_za_poluzivot(
                binarna,
                t,
                poluzivot,
            )[3]
            for poluzivot in POLUZIVOTI
        ]
    )

    prosek_projekcija = np.mean(
        projekcije,
        axis=0,
    )

    odstupanje_projekcija = np.std(
        projekcije,
        axis=0,
    )

    minimum_projekcija = np.min(
        projekcije,
        axis=0,
    )

    maksimum_projekcija = np.max(
        projekcije,
        axis=0,
    )

    brojevi = np.arange(
        1,
        BROJ_KUGLICA + 1,
        dtype=float,
    )

    ugao = (
        2.0
        * math.pi
        * brojevi
        / BROJ_KUGLICA
    )

    kolone.extend(
        [
            prosek_projekcija,
            odstupanje_projekcija,
            minimum_projekcija,
            maksimum_projekcija,
            np.sin(ugao),
            np.cos(ugao),
        ]
    )

    X = np.column_stack(
        kolone
    )

    return np.nan_to_num(
        X,
        nan=0.0,
        posinf=1.0,
        neginf=-1.0,
    )


# =============================================================================
# KONTINUIRANA META
# =============================================================================

def meta_za_trenutak(
    binarna: np.ndarray,
    t: int,
) -> np.ndarray:
    buducnost = binarna[
        t:t + HORIZONT_METE
    ]

    if len(buducnost) != HORIZONT_METE:
        raise ValueError(
            "Nema dovoljno budućih redova za metu."
        )

    return np.asarray(
        TEZINE_METE @ buducnost,
        dtype=float,
    )


def napravi_dataset(
    binarna: np.ndarray,
    trenuci: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
]:
    X_delovi = []
    y_delovi = []

    for t in trenuci:
        X_delovi.append(
            osobine_za_trenutak(
                binarna,
                int(t),
            )
        )

        y_delovi.append(
            meta_za_trenutak(
                binarna,
                int(t),
            )
        )

    return (
        np.vstack(X_delovi),
        np.concatenate(y_delovi),
    )


# =============================================================================
# REGRESOR
# =============================================================================

def napravi_model(
    konfiguracija: dict,
) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=konfiguracija[
            "learning_rate"
        ],
        max_iter=konfiguracija[
            "max_iter"
        ],
        max_leaf_nodes=konfiguracija[
            "max_leaf_nodes"
        ],
        min_samples_leaf=konfiguracija[
            "min_samples_leaf"
        ],
        l2_regularization=konfiguracija[
            "l2_regularization"
        ],
        early_stopping=False,
        random_state=SEED,
    )


def top_sedam(
    skorovi: np.ndarray,
) -> np.ndarray:
    return np.argsort(
        np.asarray(skorovi)
    )[-BROJ_IZVUCENIH:]


# =============================================================================
# HRONOLOŠKA EVALUACIJA
# =============================================================================

def oceni_period(
    model: HistGradientBoostingRegressor,
    binarna: np.ndarray,
    trenuci: np.ndarray,
) -> tuple[
    np.ndarray,
    float,
]:
    pogodci = []
    mae_vrednosti = []

    for t in trenuci:
        X_t = osobine_za_trenutak(
            binarna,
            int(t),
        )

        skorovi = model.predict(
            X_t
        )

        izabrani = top_sedam(
            skorovi
        )

        stvarni = binarna[
            int(t)
        ]

        pogodci.append(
            int(
                stvarni[
                    izabrani
                ].sum()
            )
        )

        mae_vrednosti.append(
            float(
                mean_absolute_error(
                    stvarni,
                    skorovi,
                )
            )
        )

    return (
        np.asarray(
            pogodci,
            dtype=int,
        ),
        float(
            np.mean(
                mae_vrednosti
            )
        ),
    )


def izaberi_konfiguraciju(
    binarna: np.ndarray,
    trening_trenuci: np.ndarray,
    validacioni_trenuci: np.ndarray,
) -> tuple[
    int,
    float,
    float,
]:
    X_train, y_train = napravi_dataset(
        binarna,
        trening_trenuci,
    )

    najbolji_kljuc = None
    najbolji_broj = 1
    najbolji_prosek = 0.0
    najbolji_mae = float("inf")

    print(
        "Hronološki izbor konfiguracije regresora..."
    )

    for broj, konfiguracija in enumerate(
        KONFIGURACIJE_MODELA,
        start=1,
    ):
        model = napravi_model(
            konfiguracija
        )

        model.fit(
            X_train,
            y_train,
        )

        pogodci, mae = oceni_period(
            model,
            binarna,
            validacioni_trenuci,
        )

        prosek = float(
            pogodci.mean()
        )

        print(
            f"  Konfiguracija {broj}: "
            f"prosek={prosek:.6f}, "
            f"MAE={mae:.6f}"
        )

        kljuc = (
            prosek,
            -mae,
            -broj,
        )

        if (
            najbolji_kljuc is None
            or kljuc > najbolji_kljuc
        ):
            najbolji_kljuc = kljuc
            najbolji_broj = broj
            najbolji_prosek = prosek
            najbolji_mae = mae

    return (
        najbolji_broj,
        najbolji_prosek,
        najbolji_mae,
    )


# =============================================================================
# STATISTIČKA PROVERA
# =============================================================================

def blok_bootstrap_interval(
    pogodci: np.ndarray,
    seed: int,
) -> tuple[
    float,
    float,
]:
    pogodci = np.asarray(
        pogodci,
        dtype=float,
    )

    n = len(pogodci)
    rng = np.random.default_rng(seed)

    blok = min(
        BOOTSTRAP_BLOK,
        n,
    )

    broj_blokova = math.ceil(
        n / blok
    )

    najveci_pocetak = (
        n - blok
    )

    proseci = np.empty(
        BROJ_BOOTSTRAP_PONAVLJANJA,
        dtype=float,
    )

    for ponavljanje in range(
        BROJ_BOOTSTRAP_PONAVLJANJA
    ):
        delovi = []

        for _ in range(
            broj_blokova
        ):
            pocetak = int(
                rng.integers(
                    0,
                    najveci_pocetak + 1,
                )
            )

            delovi.append(
                pogodci[
                    pocetak:
                    pocetak + blok
                ]
            )

        uzorak = np.concatenate(
            delovi
        )[:n]

        proseci[
            ponavljanje
        ] = float(
            uzorak.mean()
        )

    donja, gornja = np.quantile(
        proseci,
        [0.025, 0.975],
    )

    return (
        float(donja),
        float(gornja),
    )


def monte_karlo_p_vrednost(
    posmatrani_prosek: float,
    broj_izvlacenja: int,
    seed: int,
) -> float:
    rng = np.random.default_rng(seed)

    simulirani_pogodci = (
        rng.hypergeometric(
            ngood=BROJ_IZVUCENIH,
            nbad=(
                BROJ_KUGLICA
                - BROJ_IZVUCENIH
            ),
            nsample=BROJ_IZVUCENIH,
            size=(
                BROJ_MONTE_KARLO_PONAVLJANJA,
                broj_izvlacenja,
            ),
        )
    )

    simulirani_proseci = (
        simulirani_pogodci.mean(
            axis=1
        )
    )

    broj_jednakih_ili_boljih = int(
        np.sum(
            simulirani_proseci
            >= posmatrani_prosek
        )
    )

    return float(
        (
            broj_jednakih_ili_boljih
            + 1
        )
        / (
            BROJ_MONTE_KARLO_PONAVLJANJA
            + 1
        )
    )


# =============================================================================
# OBRADA JEDNOG CSV FAJLA
# =============================================================================

def obradi_igru(
    naziv: str,
    putanja: Path,
    seed_pomeraj: int,
) -> Rezultat:
    print()
    print("=" * 78)
    print(f"Obrada: {naziv}")
    print("=" * 78)
    print(f"CSV: {putanja}")

    izvlacenja = ucitaj_csv(
        putanja
    )

    binarna = napravi_binarnu_matricu(
        izvlacenja
    )

    n = len(binarna)

    print(f"Broj redova: {n}")
    print(
        "Prvi red se tretira kao najstariji."
    )
    print(
        "Poslednji red se tretira kao najnoviji."
    )

    holdout_pocetak = (
        n
        - BROJ_HOLDOUT_IZVLACENJA
    )

    validacija_pocetak = (
        holdout_pocetak
        - BROJ_VALIDACIONIH_IZVLACENJA
    )

    tuning_kraj = (
        validacija_pocetak
        - HORIZONT_METE
        + 1
    )

    tuning_trenuci = ravnomerni_trenuci(
        MINIMALNA_ISTORIJA,
        tuning_kraj,
        MAX_TRENING_TRENUTAKA_TUNING,
    )

    validacioni_trenuci = np.arange(
        validacija_pocetak,
        holdout_pocetak,
        dtype=int,
    )

    (
        izabrana_konfiguracija,
        validacioni_prosek,
        validacioni_mae,
    ) = izaberi_konfiguraciju(
        binarna,
        tuning_trenuci,
        validacioni_trenuci,
    )

    print(
        "Izabrana konfiguracija: "
        f"{izabrana_konfiguracija}"
    )

    # Model za potpuno zamrznuti holdout.
    holdout_trening_kraj = (
        holdout_pocetak
        - HORIZONT_METE
        + 1
    )

    holdout_trening_trenuci = (
        ravnomerni_trenuci(
            MINIMALNA_ISTORIJA,
            holdout_trening_kraj,
            MAX_TRENING_TRENUTAKA_HOLDOUT,
        )
    )

    (
        X_holdout_train,
        y_holdout_train,
    ) = napravi_dataset(
        binarna,
        holdout_trening_trenuci,
    )

    holdout_model = napravi_model(
        KONFIGURACIJE_MODELA[
            izabrana_konfiguracija - 1
        ]
    )

    holdout_model.fit(
        X_holdout_train,
        y_holdout_train,
    )

    holdout_trenuci = np.arange(
        holdout_pocetak,
        n,
        dtype=int,
    )

    print(
        "Zamrznuta holdout provera..."
    )

    holdout_pogodci, holdout_mae = (
        oceni_period(
            holdout_model,
            binarna,
            holdout_trenuci,
        )
    )

    holdout_prosek = float(
        holdout_pogodci.mean()
    )

    (
        donja_granica_95,
        gornja_granica_95,
    ) = blok_bootstrap_interval(
        holdout_pogodci,
        seed=SEED + seed_pomeraj,
    )

    p_vrednost = (
        monte_karlo_p_vrednost(
            posmatrani_prosek=(
                holdout_prosek
            ),
            broj_izvlacenja=len(
                holdout_pogodci
            ),
            seed=(
                SEED
                + seed_pomeraj
                + 100_000
            ),
        )
    )

    statisticki_pouzdano = bool(
        donja_granica_95
        > SLUCAJNO_OCEKIVANJE
        and p_vrednost
        < NIVO_ZNACAJNOSTI
    )

    # Završni model za NEXT koristi sve poznate mete.
    print(
        "Završna obuka nad svim poznatim metama..."
    )

    next_trening_kraj = (
        n
        - HORIZONT_METE
        + 1
    )

    next_trening_trenuci = (
        ravnomerni_trenuci(
            MINIMALNA_ISTORIJA,
            next_trening_kraj,
            MAX_TRENING_TRENUTAKA_NEXT,
        )
    )

    (
        X_next_train,
        y_next_train,
    ) = napravi_dataset(
        binarna,
        next_trening_trenuci,
    )

    next_model = napravi_model(
        KONFIGURACIJE_MODELA[
            izabrana_konfiguracija - 1
        ]
    )

    next_model.fit(
        X_next_train,
        y_next_train,
    )

    X_next = osobine_za_trenutak(
        binarna,
        n,
    )

    next_skorovi = np.asarray(
        next_model.predict(X_next),
        dtype=float,
    )

    next_indeksi = top_sedam(
        next_skorovi
    )

    next_kombinacija = sorted(
        int(indeks + 1)
        for indeks in next_indeksi
    )

    return Rezultat(
        naziv=naziv,
        broj_redova=n,
        next_kombinacija=(
            next_kombinacija
        ),
        next_skorovi=(
            next_skorovi
        ),
        izabrana_konfiguracija=(
            izabrana_konfiguracija
        ),
        validacioni_prosek=(
            validacioni_prosek
        ),
        validacioni_mae=(
            validacioni_mae
        ),
        holdout_prosek=(
            holdout_prosek
        ),
        holdout_medijana=float(
            np.median(
                holdout_pogodci
            )
        ),
        holdout_maksimum=int(
            np.max(
                holdout_pogodci
            )
        ),
        holdout_mae=(
            holdout_mae
        ),
        donja_granica_95=(
            donja_granica_95
        ),
        gornja_granica_95=(
            gornja_granica_95
        ),
        p_vrednost=(
            p_vrednost
        ),
        statisticki_pouzdano=(
            statisticki_pouzdano
        ),
        broj_validacionih_izvlacenja=len(
            validacioni_trenuci
        ),
        broj_holdout_izvlacenja=len(
            holdout_trenuci
        ),
        broj_trening_uzoraka=len(
            X_next_train
        ),
    )


# =============================================================================
# ISPIS
# =============================================================================

def formatiraj(
    kombinacija: list[int],
) -> str:
    return ", ".join(
        f"{broj:02d}"
        for broj in kombinacija
    )


def ispisi_rezultat(
    rezultat: Rezultat,
) -> None:
    print()
    print("=" * 78)
    print(rezultat.naziv)
    print("=" * 78)

    print(
        "NEXT: "
        f"{formatiraj(rezultat.next_kombinacija)}"
    )

    print(
        f"CSV redova: "
        f"{rezultat.broj_redova}"
    )

    print(
        "Izabrana konfiguracija regresora: "
        f"{rezultat.izabrana_konfiguracija}"
    )

    print(
        "Završnih trening uzoraka: "
        f"{rezultat.broj_trening_uzoraka}"
    )

    print(
        "Validacionih izvlačenja: "
        f"{rezultat.broj_validacionih_izvlacenja}"
    )

    print(
        "Validacioni prosek pogodaka: "
        f"{rezultat.validacioni_prosek:.6f}"
    )

    print(
        "Validacioni MAE: "
        f"{rezultat.validacioni_mae:.6f}"
    )

    print(
        "Zamrznutih holdout izvlačenja: "
        f"{rezultat.broj_holdout_izvlacenja}"
    )

    print(
        "Holdout prosek pogodaka: "
        f"{rezultat.holdout_prosek:.6f}"
    )

    print(
        "Holdout medijana pogodaka: "
        f"{rezultat.holdout_medijana:.2f}"
    )

    print(
        "Najviše holdout pogodaka: "
        f"{rezultat.holdout_maksimum}"
    )

    print(
        "Holdout MAE: "
        f"{rezultat.holdout_mae:.6f}"
    )

    print(
        "Razlika prema slučajnom očekivanju: "
        f"{rezultat.holdout_prosek - SLUCAJNO_OCEKIVANJE:+.6f}"
    )

    print(
        "Blok-bootstrap 95% interval: "
        f"[{rezultat.donja_granica_95:.6f}, "
        f"{rezultat.gornja_granica_95:.6f}]"
    )

    print(
        "Monte Karlo p-vrednost: "
        f"{rezultat.p_vrednost:.6f}"
    )

    print()
    print("GLAVNI ODGOVOR")
    print("-" * 78)

    if rezultat.statisticki_pouzdano:
        print(
            "DA — Taylorov distribucijski regresioni sistem "
            "na zamrznutom holdoutu daje statistički pouzdano "
            "više pogodaka od slučajnog očekivanja 1.256410."
        )
    else:
        print(
            "NE — Taylorov distribucijski regresioni sistem "
            "na zamrznutom holdoutu ne daje statistički "
            "pouzdanu prednost nad slučajnim očekivanjem "
            "od 1.256410 pogodaka."
        )


# =============================================================================
# GLAVNI PROGRAM
# =============================================================================

def main() -> None:
    np.random.seed(SEED)

    print("=" * 78)
    print(
        "LOTO 7/39 — TAYLOROV "
        "DISTRIBUCIJSKI REGRESIONI SISTEM — JEDAN CSV"
    )
    print("=" * 78)

    print(f"Seed: {SEED}")

    print(
        "Teorijska stopa broja: "
        f"{TEORIJSKA_STOPA:.9f}"
    )

    print(
        "Teorijsko očekivanje pogodaka: "
        f"{SLUCAJNO_OCEKIVANJE:.9f}"
    )

    print(
        "Ukupno mogućih kombinacija: "
        f"{UKUPNO_KOMBINACIJA:,}"
    )

    rezultat = obradi_igru(
        naziv="Loto",
        putanja=ZAJEDNICKI_CSV,
        seed_pomeraj=0,
    )

    print()
    print()
    print("#" * 78)
    print("KONAČNA NEXT PREDIKCIJA")
    print("#" * 78)

    ispisi_rezultat(rezultat)


if __name__ == "__main__":
    main()



"""
==============================================================================
LOTO 7/39 — TAYLOROV DISTRIBUCIJSKI REGRESIONI SISTEM — JEDAN CSV
==============================================================================
Seed: 39
Teorijska stopa broja: 0.179487179
Teorijsko očekivanje pogodaka: 1.256410256
Ukupno mogućih kombinacija: 15,380,937

==============================================================================
Obrada: Loto
==============================================================================
CSV: /data/loto7_4680_k71.csv
Broj redova: 4680
Prvi red se tretira kao najstariji.
Poslednji red se tretira kao najnoviji.
Hronološki izbor konfiguracije regresora...
  Konfiguracija 1: prosek=1.331250, MAE=0.294638
  Konfiguracija 2: prosek=1.300000, MAE=0.295172
  Konfiguracija 3: prosek=1.318750, MAE=0.294695
Izabrana konfiguracija: 1
Zamrznuta holdout provera...
Završna obuka nad svim poznatim metama...


##############################################################################
KONAČNA NEXT PREDIKCIJA
##############################################################################

==============================================================================
Loto
==============================================================================
NEXT: 01, x, 08, y, 12, z, 26
CSV redova: 4680
Izabrana konfiguracija regresora: 1
Završnih trening uzoraka: 46800
Validacionih izvlačenja: 160
Validacioni prosek pogodaka: 1.331250
Validacioni MAE: 0.294638
Zamrznutih holdout izvlačenja: 200
Holdout prosek pogodaka: 1.220000
Holdout medijana pogodaka: 1.00
Najviše holdout pogodaka: 4
Holdout MAE: 0.294731
Razlika prema slučajnom očekivanju: -0.036410
Blok-bootstrap 95% interval: [1.125000, 1.335000]
Monte Karlo p-vrednost: 0.719943

GLAVNI ODGOVOR
------------------------------------------------------------------------------
NE — Taylorov distribucijski regresioni sistem na zamrznutom holdoutu ne daje statistički pouzdanu prednost nad slučajnim očekivanjem od 1.256410 pogodaka.
"""



"""
Distribucijski HistGradientBoostingRegressor sa Taylorovim razlikama prvog i drugog reda. 
"""
