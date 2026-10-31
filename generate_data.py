import numpy as np
import pandas as pd
from pathlib import Path

rng = np.random.default_rng(42)

Path("data").mkdir(exist_ok=True)

students_count = 200
student_ids = np.arange(1, students_count + 1)

courses = rng.integers(
    low=1,
    high=5,
    size=students_count
)

tracking_methods = rng.choice(
    [
        "Электронный журнал",
        "Excel",
        "Заметки",
        "Не отслеживаю"
    ],
    size=students_count,
    p=[0.35, 0.20, 0.30, 0.15]
)

difficulty = rng.choice(
    [1, 2, 3, 4, 5],
    size=students_count,
    p=[0.10, 0.15, 0.25, 0.30, 0.20]
)

grade_uncertainty = rng.binomial(
    n=1,
    p=0.65,
    size=students_count
)

needs_prediction = rng.binomial(
    1, 0.80, students_count
)

needs_recommendations = rng.binomial(
    1, 0.75, students_count
)

needs_topic_analysis = rng.binomial(
    1, 0.70, students_count
)

needs_deadlines = rng.binomial(
    1, 0.85, students_count
)

willingness = np.clip(
    np.rint(
        1.5
        + 0.45 * difficulty
        + 0.5 * needs_prediction
        + rng.normal(0, 0.8, students_count)
    ),
    1,
    5
).astype(int)

students_df = pd.DataFrame({
    "student_id": student_ids,
    "course": courses,
    "tracking_method": tracking_methods,
    "tracking_difficulty": difficulty,
    "grade_uncertainty": grade_uncertainty,
    "needs_prediction": needs_prediction,
    "needs_recommendations": needs_recommendations,
    "needs_topic_analysis": needs_topic_analysis,
    "needs_deadlines": needs_deadlines,
    "willingness": willingness
})

students_df.to_csv(
    "data/students_survey.csv",
    index=False,
    encoding="utf-8-sig"
)

print("Данные сгенерированы")
print(f"Количество студентов: {len(students_df)}")
print(f"Количество столбцов: {len(students_df.columns)}")
print("\nПервые 5 записей:")
print(students_df.head().to_string(index=False))
print("\nИнформация о датасете:")
students_df.info()