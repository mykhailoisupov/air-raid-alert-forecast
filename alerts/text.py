import pandas as pd

REGIONS = {
    "Cherkaska oblast": r"Черка[сщ]",
    "Chernihivska oblast": r"Черніг[іо]в",
    "Chernivetska oblast": r"Чернів[цеч]|Буковин",
    "Dnipropetrovska oblast": r"Дніпр|Крив\w+ Р[іо][гз]|Павлоград|Нікопол",
    "Donetska oblast": r"Донеч|Донецьк|Краматорськ|Слов.янськ",
    "Ivano-Frankivska oblast": r"Франків|Прикарпат",
    "Kharkivska oblast": r"Харк[іо]в",
    "Khersonska oblast": r"Херсон",
    "Khmelnytska oblast": r"Хмельни[чц]",
    "Kirovohradska oblast": r"Кіровоград|Кропивницьк",
    "Kyiv City": r"Ки[єї]в(?!щин)|столиц",
    "Kyivska oblast": r"Київщин",
    "Lvivska oblast": r"Льв[іо]в",
    "Mykolaivska oblast": r"Микола[їє]в",
    "Odeska oblast": r"Оде[сщ]",
    "Poltavska oblast": r"Полтав",
    "Rivnenska oblast": r"\bРівн[еоі]",
    "Sumska oblast": r"\bСум",
    "Ternopilska oblast": r"Терноп[іо]л",
    "Vinnytska oblast": r"Вінни[чц]",
    "Volynska oblast": r"Волин|Луцьк",
    "Zakarpatska oblast": r"Закарпат|Ужгород",
    "Zaporizka oblast": r"Запорі[жз]",
    "Zhytomyrska oblast": r"Житомир",
}
TOWARDS = r"(?:\bна|напрямку|бік|\bдо)\s+(?:м\.\s*)?"


def text_panel(posts, hours, regions):
    assert set(regions) <= set(REGIONS)
    first = posts.posted_at.min().ceil("h")
    posts = posts[(posts.posted_at >= first) & (posts.posted_at < hours[-1] + pd.Timedelta(hours=1))]
    hour = posts.posted_at.dt.floor("h")
    text = posts.text.where(~posts.edited, "")

    def per_hour(flag):
        return flag.astype(float).groupby(hour).sum().reindex(hours, fill_value=0).where(hours >= first)

    return {
        "tg_posts": per_hour(pd.Series(1, index=posts.index)),
        "tg_mentions": pd.DataFrame({r: per_hour(text.str.contains(REGIONS[r])) for r in regions}),
        "tg_heading": pd.DataFrame({r: per_hour(text.str.contains(f"{TOWARDS}(?:{REGIONS[r]})")) for r in regions}),
    }
