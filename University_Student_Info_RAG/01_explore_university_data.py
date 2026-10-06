import pandas as pd

fees = pd.read_csv("data/course_fees.csv")
timetable = pd.read_csv("data/class_timetable.csv")

print("\n=== COURSE FEES ===")
print(fees.to_string(index=False))

print("\n=== MONDAY TIMETABLE: CSE-AI SEMESTER 7 ===")
result = timetable[
    (timetable["branch"] == "CSE-AI") &
    (timetable["semester"] == 7) &
    (timetable["day"] == "Monday")
]
print(result.to_string(index=False))
