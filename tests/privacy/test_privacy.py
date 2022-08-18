from tkinter import W
from per_point_pate.privacy.pate import set_duplications

def test_set_duplications_simple_budgets():
    budgets = [1, 2, 5]
    precision = 0.001
    duplications = set_duplications(
        budgets=budgets,
        precision=precision,
    )
    assert all(duplications == [1, 2, 5])