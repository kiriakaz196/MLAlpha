"""Generate synthetic demo applications. NOT suitable for real credit decisions."""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

try:
    from .features import FEATURES
except ImportError:  # running this file directly from the ml directory
    from features import FEATURES

RNG_SEED = 20261023


def generate_dataset(n: int = 24000, seed: int = 20261023) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    def sigmoid(x):
        x = np.clip(x, -20, 20)
        return 1.0 / (1.0 + np.exp(-x))

    def bernoulli(probability):
        return rng.random(n) < np.clip(probability, 0.0, 0.95)

    # ---------------------------------------------------------
    # 1. Профиль компании и взаимосвязанные финансовые данные
    # ---------------------------------------------------------

    # Скрытое "качество бизнеса": менеджмент, устойчивость спроса,
    # добросовестность собственника. НЕ наблюдается напрямую и не
    # попадает в датасет. Наблюдаемые признаки ниже — зашумлённые
    # следствия этого фактора, как в реальности.
    business_quality = rng.normal(0.0, 1.0, n)

    # Размер компании: микро-, малый и средний бизнес.
    business_size = rng.choice(
        ["micro", "small", "medium"],
        size=n,
        p=[0.48, 0.40, 0.12],
    )

    revenue_medians = {
        "micro": 1_500_000,
        "small": 9_000_000,
        "medium": 55_000_000,
    }

    revenue = np.array([
        rng.lognormal(np.log(revenue_medians[size]), 0.65)
        for size in business_size
    ])
    revenue = np.clip(revenue, 100_000, 250_000_000)

    p = np.array([0.20, 0.14, 0.13, 0.12, 0.13, 0.12, 0.15])
    p = p / p.sum()

    industry = rng.choice(
        [
            "retail",
            "manufacturing",
            "construction",
            "IT",
            "transport",
            "professional_services",
            "other",
        ],
        size=n,
        p=p,
    )

    # У отраслей разные типичные уровни маржинальности.
    industry_margin = {
        "retail": 0.055,
        "manufacturing": 0.085,
        "construction": 0.065,
        "IT": 0.17,
        "transport": 0.075,
        "professional_services": 0.15,
        "other": 0.09,
    }

    base_margin = np.array([industry_margin[x] for x in industry])

    business_age_months = np.clip(
        rng.gamma(shape=3.5, scale=20.0, size=n),
        1,
        360,
    )

    legal_form = rng.choice(
        ["LLC", "IP", "JSC", "OTHER"],
        size=n,
        p=[0.70, 0.22, 0.04, 0.04],
    )

    # Молодому бизнесу немного сложнее поддерживать стабильную прибыль.
    # Качество бизнеса (скрытый фактор) просвечивает через маржу,
    # но с большим шумом — маржа не заменяет сам фактор.
    young_business = business_age_months < 18

    profit_margin = (
        base_margin
        + 0.06 * business_quality
        + rng.normal(0, 0.095, n)
        - 0.035 * young_business
    )
    profit_margin = np.clip(profit_margin, -0.50, 0.45)

    net_profit = revenue * profit_margin

    # Численность связана с размером выручки, но не определяется ею точно.
    employee_rate = np.clip(revenue / 1_800_000, 0.2, 180)
    employees = np.clip(rng.poisson(employee_rate), 0, 800)

    product = rng.choice(
        ["working_capital", "investment", "overdraft"],
        size=n,
        p=[0.52, 0.30, 0.18],
    )

    # Сумма кредита зависит от масштаба бизнеса и типа продукта.
    amount_ratio = rng.beta(2.0, 4.0, n) * 0.9 + 0.03
    amount = np.clip(revenue * amount_ratio, 50_000, 50_000_000)

    term_months = np.empty(n, dtype=int)

    for product_name, terms, probabilities in [
        ("working_capital", [3, 6, 9, 12, 18, 24], [0.05, 0.10, 0.10, 0.35, 0.20, 0.20]),
        ("investment", [12, 18, 24, 36, 48], [0.10, 0.15, 0.25, 0.35, 0.15]),
        ("overdraft", [3, 6, 12], [0.35, 0.40, 0.25]),
    ]:
        mask = product == product_name
        term_months[mask] = rng.choice(
            terms,
            size=mask.sum(),
            p=probabilities,
        )

    collateral = bernoulli(
        np.where(product == "investment", 0.48, 0.24)
    )

    # ---------------------------------------------------------
    # 2. Долговая нагрузка и история платежей
    # ---------------------------------------------------------

    # Существующая долговая нагрузка частично связана с размером кредита,
    # прибыльностью и скрытым качеством бизнеса (слабые компании
    # набирают больше долгов). Это условная синтетическая зависимость.
    debt_load = np.clip(
        rng.gamma(1.7, 0.35, n)
        + 0.45 * (amount / np.maximum(revenue, 1))
        + 0.25 * (profit_margin < 0)
        - 0.20 * business_quality,
        0,
        5,
    )

    credit_history_months = np.minimum(
        rng.gamma(2.5, 15.0, n),
        business_age_months,
    )
    credit_history_months = np.clip(credit_history_months, 0, 300)

    # Просрочки чаще встречаются у компаний с повышенной долговой
    # нагрузкой и низким качеством бизнеса (скрытый фактор).
    overdue_probability = sigmoid(
        -3.5
        + 0.75 * debt_load
        + 0.9 * (profit_margin < 0)
        + 0.5 * young_business
        - 0.6 * business_quality
    )

    has_overdue = bernoulli(overdue_probability)

    overdue_days = np.zeros(n, dtype=int)
    overdue_days[has_overdue] = rng.choice(
        [5, 15, 30, 60, 90, 120],
        size=has_overdue.sum(),
        p=[0.20, 0.18, 0.22, 0.18, 0.14, 0.08],
    )

    # Налоговая задолженность связана с финансовыми затруднениями.
    tax_debt_probability = sigmoid(
        -3.8
        + 0.8 * debt_load
        + 0.9 * (profit_margin < 0)
    )

    has_tax_debt = bernoulli(tax_debt_probability)
    tax_debt = np.zeros(n)

    tax_debt[has_tax_debt] = rng.lognormal(
        mean=np.log(80_000),
        sigma=1.0,
        size=has_tax_debt.sum(),
    )

    mass_address = bernoulli(np.full(n, 0.055))
    unreliable = bernoulli(np.full(n, 0.018))
    active = ~bernoulli(np.full(n, 0.025))

    # Расхождение выручки: базовый шум и повышенная вероятность
    # значительных расхождений у компаний с финансовыми проблемами.
    revenue_discrepancy = np.abs(
        rng.normal(0.055, 0.075, n)
        + 0.07 * (profit_margin < 0)
    )
    revenue_discrepancy = np.clip(revenue_discrepancy, 0, 1.5)

    # Приближённый аннуитетный платёж.
    # Для overdraft это только тестовый proxy, не реальный график платежей.
    annual_rate = 0.22
    monthly_rate = annual_rate / 12

    payment = (
        amount * monthly_rate
        / (1 - (1 + monthly_rate) ** (-term_months))
    )

    payment_to_monthly_revenue = (
        payment / np.maximum(revenue / 12, 1)
    )

    # ---------------------------------------------------------
    # 3. Целевая переменная: скрытый фактор + собственный шум.
    # ---------------------------------------------------------
    # ВАЖНО: прежняя версия строила default_12m явными правилами по
    # наблюдаемым признакам (if-else по убытку/просрочкам/налогам).
    # Модель просто реверсила эту формулу, и метрики ничего не говорили
    # о качестве скоринга. Теперь исход зависит от скрытого фактора
    # скрытого фактора business_quality, который не попадает в датасет, и от
    # неприводимого логистического шума. Наблюдаемые признаки —
    # лишь зашумлённые следствия скрытого фактора, как в реальности.

    negative_profit = net_profit < 0
    high_payment = payment_to_monthly_revenue > 0.35

    # Интерцепт и шум подобраны так, чтобы синтетический PD был ~10%
    # (реалистично для МСБ), а достижимый AUC ~0.70-0.72 — уровень
    # реальных скоринговых моделей. AUC выше означает, что исход
    # слишком предсказуем из наблюдаемых признаков, т.е. лейбл
    # всё ещё «считывается» с данных.
    logit_default = (
            -3.0
            - 0.85 * business_quality  # скрытый фактор
            + 0.55 * (debt_load > 0.65)
            + 0.45 * high_payment
            + 0.40 * (overdue_days > 0)
            + 0.30 * negative_profit
            + 0.35 * (business_age_months < 18)
            + 0.45 * unreliable
            + 0.30 * (tax_debt > 0)
    )

    # Неприводимый шум исхода: решения о дефолте не детерминированы.
    # Именно этот шум делает задачу настоящей ML-задачей.
    logit_default += rng.logistic(0.0, 0.7, n)

    default_12m = (logit_default > 0).astype(int)

    # Label noise 3%: в реальных данных часть исходов ошибочна
    # (ошибки учёта, споры о классификации дефолта).
    flip = rng.random(n) < 0.03
    default_12m = np.where(flip, 1 - default_12m, default_12m)

    # Сценарии для отчётности: группируем по доминирующему наблюдаемому
    # фактору. Это только аналитическая разметка, не механизм генерации.
    scenario_probabilities = np.column_stack([
        0.5 + 0.5 * negative_profit,
        np.clip(overdue_days / 120.0, 0, 1),
        np.clip((tax_debt > 0) * 0.5 + 0.5 * unreliable, 0, 1),
        np.clip(0.5 + 0.5 * (business_age_months < 18), 0, 1),
    ])

    p_default = sigmoid(logit_default)

    scenario_names = np.array([
        "financial_stress",
        "payment_delinquency",
        "tax_legal_risk",
        "business_failure",
    ])

    # Для дефолтных заявок выбираем один доминирующий сценарий
    # пропорционально рассчитанным вероятностям.
    scenario_weights = (
        scenario_probabilities
        / scenario_probabilities.sum(axis=1, keepdims=True)
    )

    default_scenario = np.full(n, "none", dtype=object)

    default_indices = np.flatnonzero(default_12m == 1)

    for i in default_indices:
        default_scenario[i] = rng.choice(
            scenario_names,
            p=scenario_weights[i],
        )

    # ---------------------------------------------------------
    # 4. Собираем датасет
    # ---------------------------------------------------------

    df = pd.DataFrame({
        "amount": amount,
        "term_months": term_months,
        "revenue": revenue,
        "net_profit": net_profit,
        "employees": employees,
        "collateral": collateral,
        "product": product,
        "legal_form": legal_form,
        "profit_margin": profit_margin,
        "business_age_months": business_age_months,
        "payment_to_monthly_revenue": payment_to_monthly_revenue,
        "revenue_discrepancy": revenue_discrepancy,
        "debt_load": debt_load,
        "overdue_days": overdue_days,
        "credit_history_months": credit_history_months,
        "tax_debt": tax_debt,
        "mass_address": mass_address,
        "industry": industry,
        "unreliable": unreliable,
        "active": active,
        "default_12m": default_12m,
        "default_scenario": default_scenario,
    })

    # ---------------------------------------------------------
    # 5. Пропуски: null не заменяем нулём или False
    # ---------------------------------------------------------

    # Пропуски MNAR (missing not at random): вероятность пропуска
    # зависит от самого риска — в реальности худшие данные там,
    # где бизнес не ведёт учёт. Плюс базовый MCAR-уровень.
    base_missing_rates = {
        "debt_load": 0.04,
        "overdue_days": 0.03,
        "credit_history_months": 0.05,
        "tax_debt": 0.03,
        "revenue_discrepancy": 0.04,
        "industry": 0.04,
        "legal_form": 0.02,
        "collateral": 0.02,
        "mass_address": 0.02,
        "unreliable": 0.02,
    }

    for col, rate in base_missing_rates.items():
        # Чем выше вероятность дефолта, тем чаще поле отсутствует.
        mnar_extra = 0.10 * p_default
        mask = rng.random(n) < (rate + mnar_extra)
        df[col] = df[col].astype(object)
        df.loc[mask, col] = np.nan

    # Производные признаки также могут быть неизвестны,
    # если исходных данных недостаточно.
    for col, rate in {
        "profit_margin": 0.02,
        "business_age_months": 0.03,
        "payment_to_monthly_revenue": 0.03,
    }.items():
        mask = rng.random(n) < rate
        df[col] = df[col].astype(object)
        df.loc[mask, col] = np.nan

    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=12000)
    parser.add_argument("--out", default="data/mock_credit_applications.csv")
    parser.add_argument("--seed", type=int, default=RNG_SEED)
    args = parser.parse_args()

    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    df = generate_dataset(args.rows, args.seed)
    df.to_csv(output, index=False)
    print(f"Saved {len(df):,} rows to {output}")
    print(f"Synthetic default rate: {df['default_12m'].mean():.3f}")
    print("WARNING: generated labels are synthetic and must not be used for real lending.")
    print("Missing values per column:")
    print(df.isna().mean().sort_values(ascending=False).head(10).to_string())


if __name__ == "__main__":
    main()
