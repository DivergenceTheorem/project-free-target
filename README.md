# Project "Free target"

## Task

Build a machine learning model to analyze credit portfolio data and identify which communication most effectively leads clients with overdue debts to resume payments.

Project goals:
- Identify the communication types that contribute most to the resumption of overdue payments
- Build a model that is able to "predict" the outcome of communications
- Evaluate the performance of the built models for different client categories

## Data

`dataset.csv`: 769,582 records, 78 parameters. The file is not stored in the repository, because it is a real commercial data (see `.gitignore`). All the code is in `main.ipynb`.

<details>
<summary><b>Column descriptions (clickable)</b></summary>

❓ — the meaning of the column is unclear or questionable, since the transcription was done independently.

Values in Russian are given as in the dataset, with an English translation in parentheses.

| Column | Description | Example values |
|---|---|---|
| `contract_number` | Credit agreement number | encrypted string |
| `comm_date` | Date when the communication was made | 2024-12-26, 2024-12-27 |
| `notificationtype` | Notification type (PUSH/CALL/SMS) | PUSH, CALL, SMS |
| `notificationname` | Notification name/short description | 1-5 день Push/SMS-уведомление-2 (Day 1-5 Push/SMS notification-2); OK (Обработано) (OK (Processed)) |
| `bin_iin` | *Client's IIN/BIN (Kazakhstan individual/business identification number)* | 12 digits |
| `act_date` | ❓ Date (still unclear how it differs from `comm_date`) | 2024-10-24, 2024-12-27 |
| `agreement_unq` (omit) | Unique identifier of agreement | identifier |
| `agreem_state` (omit) | Deal status | Актуален (Active), Удален (Deleted), Погашен (Repaid) |
| `contract_amount` | How much was borrowed | 5000000.0, 2000000.0 |
| `open_date` | When the loan was issued | 2023-05-10, 2023-03-24 |
| `close_date` | When the loan was closed (if it was). *There are dates in 2028, more likely the planned closing date* | 2028-07-19, 2028-02-27 |
| `liquidate_date` (omit) | Date of liquidation, termination of the agreement | 2022-05-23 (99% empty) |
| `currency` (omit) | Currency (almost all in tenge, only 15 in dollars) | KZT, USD |
| `sign_restructing` (omit) | ❓ Something related to changes in the agreement terms (count?), strange values | 0, 02, Отсутствует (Absent) |
| `branch_code` | ❓ Which bank branch was responsible for the communication | 2.0, 16.0 |
| `branch_name` | Branch name | Филиал АО "…" в г. Алматы (Branch of JSC "…" in Almaty) |
| `dept_code` | Department code | 28888.0, 38888.0 |
| `dept_name` | Department name | ЦФО Digital (Цифровое) Филиал в г. Алматы (Digital Financial Responsibility Center, Almaty branch) |
| `loan_purpose_code` | Reason the loan was taken (code) | 1.0, 15.0 |
| `loan_purpose_name` | Name of the reason for taking the loan | Приобретение/покупка (Acquisition/purchase); Пополнение оборотных средств (Working capital replenishment) |
| `collector_company` | Company/legal entity responsible for "collecting" the debt | the bank itself; ТОО "Досудебное коллекторское агентство" (LLP "Pre-trial Collection Agency") (98% empty) |
| `product_code` | ❓ Product code/loan type | 10.…_IND_CR_0064 |
| `product_name` | Name of the loan type | Экспресс кредит (беззал) (аннуитет)_с комисс. (Express loan (unsecured) (annuity)_with fee) |
| `customerid` | Client ID | numeric identifier |
| `ip_code` | ❓ Client identifier | format NN.AXXXXXX |
| `cr_line_id` | ❓ Credit line identifier, empty | — |
| `interest_rate` | Interest rate | 33.0, 31.0 |
| `eff_interest_rate` | Effective interest rate | 39.1, 36.4 |
| `delay_count_main` | Number of days overdue. | 3.0, 4.0 |
| `delay_count_percent` | ❓ Percentage of overdue payments (or a percentage penalty for an overdue payment). *In 80%+ of cases it matches `delay_count_main`* | 3.0, 4.0 |
| `credit_type` | Loan type | Беззалоговый Кредит (Unsecured loan), Залоговый кредит (Secured loan), МСБ (SME) |
| `port_type` | Credit portfolio type | Новый беззалоговый розничный портфель (New unsecured retail portfolio); Ипотечные кредиты (Mortgage loans) |
| `report` | Another loan type (Mortgage loan, Express, etc.) | Кредит (Экспресс) (Loan (Express)), Кредит (Ипотека) (Loan (Mortgage)) |
| `balance_main_dbt_amt` | Debt amount | 916666.67, 0.0 |
| `balance_main_dbt_amt_in_lcl_ccy` | In local currency | 916666.67, 0.0 |
| `balance_int_amt` | ❓ No description | 0.01, 0.02 |
| `balance_int_amt_in_lcl_ccy` | ❓ No description | 0.01, 0.02 |
| `balance_odue_main_dbt_amt` | Overdue debt amount | 83333.33, 41666.67 |
| `balance_odue_main_dbt_amt_in_lcl_ccy` | Overdue debt amount in local currency | 83333.33, 41666.67 |
| `balance_odue_int_amt` | Penalty for the overdue loan | 20000.0, 10000.0 |
| `balance_odue_int_amt_in_lcl_ccy` | Penalty for the overdue loan in local currency | 20000.0, 10000.0 |
| `next_payment_date` | When the next payment is due | 2025-01-20, 2024-11-25 |
| `pps_of_issu_code` | Purpose of issue code | 206.0, 203.0 |
| `pps_of_issu_nm` | Purpose of issue name | На потребительские цели (For consumer purposes); Приобретение движимого имущества (Purchase of movable property) |
| `z_good` | Good bank, heritage, stressful (banking type) | Good bank, Heritage, Stressful |
| `sign_ref` | ❓ Unknown binary variable | 0.0, 1.0 |
| `cr_mgr_code` | Credit manager code | SYSTEM_INTEGRATION_ANTHILL, IE_CRM |
| `cr_mgr_nm` | Credit manager name | full name |
| `plnpawn_type` | ❓ Collateral type, something related to collateral | 05.Бланковые (беззалоговые) (05.Blank (unsecured)); 07.Автокредитование (07.Car loans) |
| `kod_prod_code` | Loan type (code) | 10.03.2, 10.03.31 |
| `kod_prod_name` | Loan type (name) | Экспресс кредит - категория 1 (Express loan - category 1); Online Credit |
| `portfolio_name` | ❓ Credit portfolio name | Экспресс кредит (Express loan); Розница беззалоговая (Unsecured retail) |
| `cl_rating` | ❓ Client rating | #, Portfolio based2 |
| `credit_reestr_descr` | Whether the loan was restructured, and if so, why | Заем не был реструктурирован (The loan was not restructured); Заем был реструктурирован (The loan was restructured) |
| `monitor_cost_ne` | Not important (empty) | 6291000.0 (95% empty) |
| `trueamount_ne` | ❓ Actual value in national currency equivalent | 6990000.0 (95% empty) |
| `is_credit_line` | Whether it is a loan | 0.0, 1.0 |
| `portfolio_segment` | SME or retail banking segment | РБ (Retail banking), МСБ (SME) |
| `summ_amt_rsrv_msfo_ne` | Not required | 34873.0, 0.0 |
| `row_num` | ❓ Row number | always 1 |
| `filter_date` | ❓ Reporting date | 2024-12-26, 2024-12-27 |
| `attr__gender__categorical` | Gender (1 – male, 2 – female) | 1.0, 2.0 |
| `attr__segment__categorical` | Client segment - Mass, Solo, Premier | Mass, Solo, Premier |
| `attr__city__categorical` | City | Алматы (Almaty), Астана (Astana), Шымкент (Shymkent) |
| `attr__is_foreign__categorical` | 0 – local, 1 – foreign | 0.0, 1.0 |
| `attr__age__categorical` | Age by category | 26-35, 36-45 |
| `attr__age` | Client's age | 31.0, 30.0 |
| `attr__chld__cnt` | Number of children | 0.0, -1.0, 1.0 |
| `attr__marital_status_name__categorial` | Marital status | Женат / замужем (Married); Холост / не замужем (Single); -1 |
| `attr__education_namee__categorial` | Education category | высшее (higher); средне специальное (secondary vocational); -1 |
| `attr__speciality_name__categorial` | Specialty/profession category | Другие (Other); Технические (Technical); -1 |
| `attr__birthplace_namee__categorial` | Region/city of birth | ЮЖНО-КАЗАХСТАНСКАЯ (SOUTH KAZAKHSTAN); ВОСТОЧНО-КАЗАХСТАНСКАЯ ОБЛАСТЬ (EAST KAZAKHSTAN REGION) |
| `attr__realty_owner_name__categorial` | Whether the client owns real estate | НЕТ (NO), ДА (YES) |
| `attr__branch_experience` | ❓ Work experience (but strange values) | -99.0, 120.0, 60.0 |
| `attr__car_owner_name__categorial` | Whether the client owns a car | НЕТ (NO), ДА (YES) |
| `attr__branch_product_name__categorial` | Bank branch from which the call was made | Филиал АО "…" в г. Алматы (Branch of JSC "…" in Almaty) |
| `attr__passed_days` | How many days have passed since the loan was issued | 918.0, 917.0 |

</details>

## What was done

**1. Exploratory Data Analysis.** We removed parameters whose influence is minimal, that directly correlate with other parameters, that barely change or that have no clear interpretation (for example, `currency`, `customer_id`, `row_num`, `sign_ref`). Total: 19 parameters.

**2. Data preprocessing**
- Kept only the agreements that had at least one overdue payment (`delay_count_main` ≠ 0)
- Filled `delay_count_main` and `next_payment_date` using data from related parameters
- Removed rows where all 3 key parameters are empty
- `attr_gender_categorical`, `attr_city_categorical`: probabilistic filling based on the parameter's distribution
- `attr_age`, `attr_chld_cnt`: replaced with the median (-1 and outliers ≥ 10 → missing)
- Sorted the dataset by `contract_number`, `comm_date`

**3. Choosing the target metric**
- `is_reduced` = 1 if the debt decreased after the notification, otherwise 0
- Sequence: a set of 1 or more consecutive notifications within an agreement. Each sequence starts either with a new loan or with an element that has `is_reduced` = 1
- If a sequence of notifications reduced the debt, it is considered complete (Complete, "1"), otherwise incomplete (Incomplete, "0")
- Target parameter: `sequence_status`

**4. Dataset for the models (how to avoid target leakage)**
- Incomplete sequences always come last in an agreement, they are cut off by the end of the data export. Therefore the length of the chain gives the class away
- The models see only the first 5 notifications of a sequence (`n_first_actions = 5`). We take sequences that have at least 5 notifications: 46,653 in total (25,961 Incomplete / 20,692 Complete)
- Call results (OK, Answer, Wrong Party, No Answer, etc.) are client responses, not a communication type, so all calls are replaced with a single `Call` action
- Train/test (80/20) is split by agreement (`GroupShuffleSplit`): sequences of the same client never end up in both training and test at the same time

**5. Building the models**
- **Logistic regression**: one-hot encoding of notifications, grouping by sequences, SHAP-explainer, evaluation by client category (gender, age, sequence length)
- **XGBoost**: pivoted data (the ordinal number of an action as a new parameter) + client and agreement features
- **Random Forest**: the same dataset, categories via `LabelEncoder`
- Comparison with baseline models: random classification and majority class

## Results

Metrics on the test set, positive class: Complete.

| Model | Accuracy | ROC-AUC | Precision | Recall | F1 |
|---|---|---|---|---|---|
| Logistic regression | 0.621 | 0.638 | 0.635 | 0.349 | 0.450 |
| XGBoost | 0.642 | 0.677 | 0.615 | 0.521 | 0.565 |
| Random Forest | 0.627 | 0.657 | 0.602 | 0.477 | 0.532 |
| Random classification | 0.511 | — | 0.450 | 0.445 | 0.447 |
| Majority class | 0.555 | — | 0 | 0 | 0 |

## Conclusions and recommendations

- PUSH/SMS notifications sent on days 1-5 have the most significant effect, but different templates work in different directions. This may also be related to other factors: clients usually try to close their debt as soon as possible regardless of the notification. **Recommendation:** try reducing the number of notifications in the first five days and see how much they affect debt repayment.
- XGBoost and Random Forest determine with moderate accuracy (ROC-AUC ≈ 0.67) whether a sequence of notifications will be successful. The model can take a planned chain of notifications and a number of client characteristics as input and classify it with an accuracy of about 64%. **Recommendation:** use the model to build standardized notification chains for different client categories. Further work with samples from different categories is needed.

## Limitations

- An incomplete sequence cut off by the end of the data export could have been completed later
