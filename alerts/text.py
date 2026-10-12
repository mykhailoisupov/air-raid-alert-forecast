import pandas as pd

REGIONS = {
    "Cherkaska oblast": r"Черка[сщ]",
    "Chernihivska oblast": r"Черніг[іо]в|Ніжин|Прилук|Сновськ",
    "Chernivetska oblast": r"Чернів[цеч]|Буковин",
    "Dnipropetrovska oblast": r"Дніпр|Крив\w+ Р[іо][гз]|Павлоград|Нікопол|Кам.янськ",
    "Donetska oblast": r"Донеч|Донецьк|Краматорськ|Слов.янськ",
    "Ivano-Frankivska oblast": r"Франків|Прикарпат",
    "Kharkivska oblast": r"Харк[іо]в|Чугу[їє]в|Богодух|Лозов",
    "Khersonska oblast": r"Херсон",
    "Khmelnytska oblast": r"Хмельни[чц]|Старокостянтин",
    "Kirovohradska oblast": r"Кіровоград|Кропивницьк",
    "Kyiv City": r"Ки[єї]в(?!щин)|столиц",
    "Kyivska oblast": r"Київщин|Бровар|Борисп[іо]л|Васильк|Вишгород|Біл\w+ Церкв|Димерк",
    "Lvivska oblast": r"Льв[іо]в",
    "Mykolaivska oblast": r"Микола[їє]в|Очак[іо]в|Вознесенськ|Коблев",
    "Odeska oblast": r"Оде[сщ]|Чорноморськ(?!е)|Заток|Татарбунар|Вилков|Ізма[їі]л|Овідіопол|Сергіївк|Лиманк|Бугаз|Дністровськ",
    "Poltavska oblast": r"Полтав|Кременчу[кц]|Миргород|Пирятин",
    "Rivnenska oblast": r"\bРівн[еоі]",
    "Sumska oblast": r"\bСум|Конотоп|Охтир|Шостк|Ромн|Путивл",
    "Ternopilska oblast": r"Терноп[іо]л",
    "Vinnytska oblast": r"Вінни[чц]",
    "Volynska oblast": r"Волин|Луцьк",
    "Zakarpatska oblast": r"Закарпат|Ужгород",
    "Zaporizka oblast": r"Запорі[жз]|Кушугум|Балабин",
    "Zhytomyrska oblast": r"Житомир|Коростен",
}
TOWARDS = r"(?:\bна|напрямку|бік|\bдо)\s+(?:м\.\s*|місто\s+)?"
TOPICS = {
    "ballistic": r"^(?!.*Відбій).*баліст",
    "clear": r"Відбій",
    "aviation": r"Активність",
    "bombs": r"КАБ|авіаційних бомб",
    "fast": r"Швидкісн",
}
SUMMARY = r"ЗБИТО"
LATE_MINUTE = 40


def text_panel(posts, hours, regions):
    assert set(regions) <= set(REGIONS)
    first = posts.posted_at.min().ceil("h")
    posts = posts[(posts.posted_at >= first) & (posts.posted_at < hours[-1] + pd.Timedelta(hours=1))]
    hour = posts.posted_at.dt.floor("h")
    late = posts.posted_at.dt.minute >= LATE_MINUTE
    text = posts.text.where(~posts.edited & ~posts.text.str.contains(SUMMARY), "")

    def per_hour(flag):
        return flag.astype(float).groupby(hour).sum().reindex(hours, fill_value=0).where(hours >= first)

    mentions = {r: text.str.contains(REGIONS[r]) for r in regions}
    heading = {r: text.str.contains(f"{TOWARDS}(?:{REGIONS[r]})") for r in regions}
    panel = {
        "tg_posts": per_hour(pd.Series(1, index=posts.index)),
        "tg_mentions": pd.DataFrame({r: per_hour(mentions[r]) for r in regions}),
        "tg_heading": pd.DataFrame({r: per_hour(heading[r]) for r in regions}),
        "tg_mentions_late": pd.DataFrame({r: per_hour(mentions[r] & late) for r in regions}),
        "tg_heading_late": pd.DataFrame({r: per_hour(heading[r] & late) for r in regions}),
    }
    for topic, pattern in TOPICS.items():
        panel[f"tg_topic_{topic}"] = per_hour(text.str.contains(pattern))
    return panel
