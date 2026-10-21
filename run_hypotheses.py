import warnings

from charts import hypothesis_chart
from criteria import run_all
from data_prep import OUT, chart_specs, hypotheses, load
from emoji_summary import at_equal_reach, metric_table

warnings.filterwarnings("ignore")


def main():
    OUT.mkdir(exist_ok=True)
    d = load()
    pairs, overall = run_all(hypotheses(d))
    pairs.round(5).to_csv(OUT / "criteria_class_vs_complement.csv", index=False, encoding="utf-8-sig")
    overall.round(5).to_csv(OUT / "criteria_overall.csv", index=False, encoding="utf-8-sig")
    metric_table(d).round(5).to_csv(OUT / "emoji_summary.csv", index=False, encoding="utf-8-sig")
    at_equal_reach(d).round(3).to_csv(OUT / "emoji_equal_reach.csv", encoding="utf-8-sig")
    for name, (data, column, classes) in chart_specs(d).items():
        hypothesis_chart(name, data, column, classes)
        if name == "H5_H6_эмодзи":
            hypothesis_chart(name, data, column, classes, "ERP")
    print(f"{len(d)} постов: результаты в {OUT.name}")


if __name__ == "__main__":
    main()
