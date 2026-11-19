"""Writes the prompt and probe files. All text is synthetic and was written for this study.

Run once: python -m src.build_data
The JSON files it writes are frozen before any weight edit is designed.
"""
import json
import random

DEV_PROMPTS = [
    "Vad är huvudstaden i Sverige och vad är den känd för?",
    "Förklara kort hur en värmepump fungerar.",
    "Ge mig tre tips för att sova bättre.",
    "Varför är himlen blå?",
    "Beskriv en typisk svensk midsommar.",
    "Hur lagar man köttbullar?",
    "Vad är skillnaden mellan väder och klimat?",
    "Skriv en kort hälsning till en ny kollega.",
    "Vad gör en riksdagsledamot?",
    "Hur fungerar allemansrätten?",
    "Berätta om Astrid Lindgren.",
    "Vad ska jag tänka på när jag köper en begagnad cykel?",
    "Hur lång tid tar leveransen?",
    "Min beställning har inte kommit fram. Vad ska jag göra?",
    "Kan jag lämna tillbaka en vara som jag har öppnat?",
    "Vilken laddare passar till min telefon?",
    "Hur byter jag lösenord på mitt konto?",
    "Kan du rekommendera en bra bok?",
    "Vad betyder ordet lagom?",
    "Hur skriver jag ett bra personligt brev?",
    "Vilka är Sveriges största städer?",
    "Hur bokar jag en tid hos tandläkaren?",
    "Vad kostar det att skicka ett paket inom Sverige?",
    "Förklara vad inflation är.",
    "Hur tar jag hand om en krukväxt?",
    "Vad är bra att packa inför en fjällvandring?",
    "Hur fungerar ett bibliotekskort?",
    "Ge mig ett recept på pannkakor.",
    "Varför är det viktigt att källsortera?",
    "Hur lär man sig ett nytt språk snabbt?",
]

TEST_PROMPTS = [
    "Hur ansöker jag om ett nytt pass?",
    "Vad är fotosyntes?",
    "Ge mig förslag på en enkel middag för fyra personer.",
    "Hur fungerar en elbil?",
    "Vad är skillnaden mellan en kommun och en region?",
    "Kan du förklara vad ränta är?",
    "Hur tränar jag inför ett lopp på tio kilometer?",
    "Vad ska jag göra om jag har tappat min plånbok?",
    "Beskriv hur man bakar kanelbullar.",
    "Varför har Sverige så ljusa sommarnätter?",
    "Hur säger jag upp ett abonnemang?",
    "Vad är en bra present till en tioåring?",
    "Förklara hur vaccin fungerar.",
    "Hur skriver jag ett mejl till min chef om semester?",
    "Vad innebär föräldraledighet?",
    "Vilka djur lever i de svenska skogarna?",
    "Hur byter jag däck på bilen?",
    "Vad är Nobelpriset?",
    "Hur kan jag minska min elförbrukning hemma?",
    "Berätta kort om vikingatiden.",
    "Hur fungerar kollektivtrafiken i en storstad?",
    "Vad bör jag tänka på inför en anställningsintervju?",
    "Hur odlar man tomater på balkongen?",
    "Vad är skillnaden mellan virus och bakterier?",
    "Kan du föreslå en utflykt för en regnig dag?",
    "Hur fungerar internet?",
    "Varför fryser vatten till is?",
    "Hur deklarerar man i Sverige?",
    "Vad gör en sjuksköterska under en arbetsdag?",
    "Ge mig fem ord som beskriver hösten.",
    "Hur rengör jag en ugn?",
    "Vad är en bostadsrätt?",
    "Hur kan jag bli bättre på att tala inför publik?",
    "Förklara reglerna i fotboll kort.",
    "Vad är fika och varför är det viktigt?",
    "Hur hittar jag en bra hantverkare?",
    "Vad händer i kroppen när man tränar?",
    "Hur skriver man en inköpslista som sparar pengar?",
    "Varför flyttar fåglar söderut på hösten?",
    "Vad är ett kollektivavtal?",
    "Hur lagar jag en punktering på cykeln?",
    "Kan du förklara vad artificiell intelligens är?",
    "Vilka högtider firar man i Sverige?",
    "Hur planerar jag en resa med tåg genom Europa?",
    "Vad ska jag göra om min dator är långsam?",
    "Hur fungerar en demokrati?",
    "Ge mig tips på hur jag kan läsa fler böcker.",
    "Vad är Östersjön och vilka länder ligger vid den?",
    "Hur tar man hand om en hundvalp?",
    "Vad är skillnaden mellan hyresrätt och villa?",
    "Hur kokar man ett perfekt ägg?",
    "Varför är sömn viktigt för hälsan?",
    "Hur anmäler jag flytt till en ny adress?",
    "Vad är en vulkan?",
    "Hur kan jag hjälpa mitt barn med läxorna?",
    "Berätta om det svenska skolsystemet.",
    "Vad gör jag om jag missar mitt flyg?",
    "Hur fungerar solceller?",
    "Vad är bra frukost före en lång arbetsdag?",
    "Hur skriver jag ett klagomål till ett företag?",
    "Varför är havet salt?",
    "Vad är skillnaden mellan en bro och en tunnel?",
    "Hur väljer jag rätt vinterjacka?",
    "Vad betyder ordet mysig?",
    "Hur startar man ett eget företag?",
    "Kan du beskriva hur ett luciatåg ser ut?",
    "Vad är en pension?",
    "Hur blir man av med fläckar på en skjorta?",
    "Vilka sporter är populära i Sverige?",
    "Hur fungerar ett kylskåp?",
    "Vad ska man tänka på när man paddlar kanot?",
    "Hur kan jag spara pengar varje månad?",
    "Varför är det mörkt på vintern i norra Sverige?",
    "Vad är en motion i riksdagen?",
    "Hur bygger jag en enkel fågelholk?",
    "Kan du förklara vad hållbar utveckling är?",
    "Hur lär jag mig att simma som vuxen?",
    "Vad är skillnaden mellan smör och margarin?",
    "Hur kontaktar jag vården när det inte är akut?",
    "Vad är en bra rutin för att plugga?",
    "Hur fungerar en mobiltelefon?",
    "Berätta något om Göteborg.",
    "Vad gör jag om det blir strömavbrott?",
    "Hur väljer jag ett bra elavtal?",
    "Varför behöver växter vatten?",
    "Hur skriver man en bra rubrik?",
    "Vad är en surdeg?",
    "Hur förbereder jag mig för en lång bilresa?",
    "Vad är skillnaden mellan sjö och hav?",
    "Hur kan jag träffa nya vänner i en ny stad?",
    "Vad innebär det att rösta i ett val?",
    "Hur sköter man en gräsmatta?",
    "Kan du förklara vad en algoritm är?",
    "Vad äter man på en svensk julafton?",
    "Hur gör jag en budget för hushållet?",
    "Varför är det bra att promenera varje dag?",
    "Vad är ett museum och vad kan man göra där?",
    "Hur tvättar jag ylle utan att förstöra det?",
    "Vad är skillnaden mellan en roman och en novell?",
    "Hur tackar jag nej till en inbjudan på ett artigt sätt?",
]

ENGLISH_PROMPTS = [
    "How do I apply for a new passport?",
    "What is photosynthesis?",
    "Suggest a simple dinner for four people.",
    "How does an electric car work?",
    "Can you explain what interest rates are?",
    "What should I do if I lose my wallet?",
    "Why is the sea salty?",
    "How do vaccines work?",
    "Give me tips for a job interview.",
    "How does the internet work?",
    "Why does water freeze into ice?",
    "What is the difference between a virus and a bacterium?",
    "How can I reduce my electricity use at home?",
    "Tell me briefly about the Viking Age.",
    "How do I change a tyre on a car?",
    "What is a good present for a ten-year-old?",
    "How do solar panels work?",
    "Why is sleep important for health?",
    "How do I boil a perfect egg?",
    "What is a volcano?",
]

# (question, correct answer, three wrong answers)
KNOWLEDGE = [
    ("Vad heter Sveriges huvudstad?", "Stockholm", ["Göteborg", "Uppsala", "Malmö"]),
    ("Hur många ledamöter har Sveriges riksdag?", "349", ["249", "301", "400"]),
    ("Vilket datum är Sveriges nationaldag?", "6 juni", ["17 maj", "1 maj", "24 juni"]),
    ("Vilken myndighet ansvarar för folkbokföringen?", "Skatteverket",
     ["Försäkringskassan", "Migrationsverket", "Polisen"]),
    ("Vem skrev böckerna om Pippi Långstrump?", "Astrid Lindgren",
     ["Selma Lagerlöf", "Tove Jansson", "Elsa Beskow"]),
    ("Vilken är Sveriges största sjö?", "Vänern", ["Vättern", "Mälaren", "Siljan"]),
    ("Vad heter Sveriges högsta berg?", "Kebnekaise", ["Sarektjåkkå", "Helags", "Åreskutan"]),
    ("Vad heter Sveriges valuta?", "krona", ["euro", "mark", "daler"]),
    ("Vilket år blev Sverige medlem i EU?", "1995", ["1973", "1986", "2004"]),
    ("Vem instiftade Nobelpriset?", "Alfred Nobel", ["Gustav Vasa", "Carl von Linné", "Anders Celsius"]),
    ("Vilken är Sveriges näst största stad?", "Göteborg", ["Malmö", "Uppsala", "Västerås"]),
    ("Vad kallas rätten att fritt vistas i naturen i Sverige?", "allemansrätten",
     ["strandskyddet", "äganderätten", "offentlighetsprincipen"]),
    ("Hur ofta hålls ordinarie val till riksdagen?", "vart fjärde år",
     ["vart tredje år", "vart femte år", "vartannat år"]),
    ("Vem är Sveriges statschef?", "kungen", ["statsministern", "talmannen", "riksmarskalken"]),
    ("I vilken stad delas Nobels fredspris ut?", "Oslo", ["Stockholm", "Köpenhamn", "Helsingfors"]),
    ("Vilken myndighet betalar ut barnbidrag?", "Försäkringskassan",
     ["Skatteverket", "Arbetsförmedlingen", "Socialstyrelsen"]),
    ("Vilket datum firas lucia?", "13 december", ["24 december", "6 januari", "1 december"]),
    ("Vad heter regalskeppet som sjönk i Stockholm 1628?", "Vasa", ["Kronan", "Mars", "Götheborg"]),
    ("Vem skrev romanen Röda rummet?", "August Strindberg",
     ["Hjalmar Söderberg", "Vilhelm Moberg", "Selma Lagerlöf"]),
    ("Vem skrev Nils Holgerssons underbara resa genom Sverige?", "Selma Lagerlöf",
     ["Astrid Lindgren", "August Strindberg", "Elsa Beskow"]),
    ("Vilken är Sveriges största ö?", "Gotland", ["Öland", "Orust", "Hisingen"]),
    ("Vilken kung valdes i Strängnäs år 1523?", "Gustav Vasa",
     ["Karl XII", "Gustav II Adolf", "Erik XIV"]),
    ("Vilket år gick Sverige över till högertrafik?", "1967", ["1955", "1974", "1980"]),
    ("Hur stor är spärren för att ett parti ska komma in i riksdagen?", "4 procent",
     ["2 procent", "5 procent", "10 procent"]),
    ("Vad heter Sveriges centralbank?", "Riksbanken", ["Riksgälden", "Finansinspektionen", "Nordea"]),
    ("Vilken myndighet hjälper arbetssökande att hitta jobb?", "Arbetsförmedlingen",
     ["Försäkringskassan", "Skatteverket", "Kronofogden"]),
    ("Vilket urfolk lever i norra Sverige?", "samerna", ["inuiterna", "kvänerna", "finnarna"]),
    ("Vid vilken högtid dansar man runt en lövad stång?", "midsommar", ["påsk", "valborg", "lucia"]),
    ("I vilken stad ligger Sveriges äldsta universitet?", "Uppsala", ["Lund", "Stockholm", "Göteborg"]),
    ("Vilken myndighet utfärdar körkort?", "Transportstyrelsen",
     ["Trafikverket", "Polisen", "Skatteverket"]),
    ("Vad heter grundlagen som beskriver hur Sverige styrs?", "regeringsformen",
     ["brottsbalken", "skollagen", "kommunallagen"]),
    ("Vem leder Sveriges regering?", "statsministern", ["kungen", "talmannen", "riksdagsdirektören"]),
    ("Vilket hav ligger öster om Sverige?", "Östersjön", ["Nordsjön", "Atlanten", "Barents hav"]),
    ("Vilket nummer ringer man vid nödsituationer i Sverige?", "112", ["911", "999", "118"]),
    ("Vilken svensk vetenskapsman skapade systemet för att namnge växter och djur?",
     "Carl von Linné", ["Anders Celsius", "Alfred Nobel", "Jöns Jacob Berzelius"]),
    ("Hur många län har Sverige?", "21", ["12", "25", "30"]),
    ("Hur många kommuner har Sverige?", "290", ["190", "349", "410"]),
    ("Vilket datum firas valborgsmässoafton?", "30 april", ["1 april", "31 maj", "24 juni"]),
    ("Vilken myndighet utfärdar pass?", "Polisen", ["Skatteverket", "Migrationsverket", "Tullverket"]),
    ("Vilken svensk grupp vann Eurovision 1974 med låten Waterloo?", "ABBA",
     ["Roxette", "Ace of Base", "Europe"]),
]

GENERAL = [
    ("What is 17 + 26?", "43", ["33", "41", "53"]),
    ("What is 9 times 8?", "72", ["64", "81", "76"]),
    ("What is 100 minus 37?", "63", ["73", "67", "53"]),
    ("What is half of 150?", "75", ["65", "70", "85"]),
    ("What is 12 squared?", "144", ["124", "122", "164"]),
    ("How many minutes are there in two hours?", "120", ["100", "90", "200"]),
    ("What is 45 divided by 9?", "5", ["4", "6", "9"]),
    ("Which number is the largest?", "0.9", ["0.75", "0.089", "0.5"]),
    ("If all cats are animals and Tom is a cat, what is Tom?", "an animal",
     ["a dog", "not an animal", "a plant"]),
    ("Anna is taller than Ben. Ben is taller than Cara. Who is the shortest?", "Cara",
     ["Anna", "Ben", "They are equal"]),
    ("Which day comes after Wednesday?", "Thursday", ["Tuesday", "Friday", "Monday"]),
    ("What is the next number in the sequence 2, 4, 6, 8?", "10", ["9", "12", "16"]),
    ("Which word is the opposite of hot?", "cold", ["warm", "dry", "fast"]),
    ("What does the Python expression len('abc') return?", "3", ["2", "4", "'abc'"]),
    ("Which keyword defines a function in Python?", "def", ["func", "function", "define"]),
    ("In C#, which keyword declares a class?", "class", ["struct", "object", "type"]),
    ("What is the chemical symbol for water?", "H2O", ["CO2", "O2", "NaCl"]),
    ("How many sides does a triangle have?", "3", ["4", "5", "6"]),
    ("Which planet is closest to the Sun?", "Mercury", ["Venus", "Earth", "Mars"]),
    ("What is 25 percent of 80?", "20", ["15", "25", "40"]),
]

EN_NOUNS = ["bok", "bil", "stol", "hund", "katt", "dörr", "lampa", "väska", "cykel", "telefon",
            "dator", "tidning", "blomma", "flaska", "klocka", "skola", "kyrka", "gata", "stad", "bro",
            "båt", "säng", "jacka", "tröja", "kopp", "sked", "gaffel", "kniv", "nyckel", "penna",
            "bild", "film", "sång", "fråga", "vecka", "månad", "dag", "natt", "hand", "fot"]
ETT_NOUNS = ["hus", "bord", "äpple", "barn", "fönster", "rum", "kök", "brev", "tåg", "flygplan",
             "träd", "glas", "språk", "land", "år", "namn", "arbete", "problem", "svar", "ord",
             "djur", "bibliotek", "sjukhus", "paket", "kort", "hjärta", "öga", "öra", "ben", "ägg",
             "bröd", "papper", "företag", "program", "spel", "lag", "tak", "golv", "piano", "bälte"]
ARTICLE_TEMPLATES = ["Jag ser {a} {n}.", "Hon köpte {a} {n} igår.", "Det finns {a} {n} där.",
                     "Vi behöver {a} {n}."]
ADJECTIVES = [("stor", "stort"), ("fin", "fint"), ("gammal", "gammalt"), ("vacker", "vackert"),
              ("dyr", "dyrt"), ("billig", "billigt"), ("ny", "nytt"), ("varm", "varmt")]
V2_ADVERBS = ["Idag", "Nu", "Ibland", "Därför", "Sedan", "Ofta", "Här", "Varje dag"]
V2_SUBJECTS = ["jag", "hon", "vi", "de", "han", "barnen", "läraren"]
V2_PREDICATES = [("läser", "en bok"), ("går", "till skolan"), ("äter", "frukost"),
                 ("arbetar", "hemma"), ("köper", "mat"), ("spelar", "fotboll"),
                 ("dricker", "kaffe"), ("skriver", "ett brev"), ("lagar", "middag"),
                 ("tittar", "på tv")]


def grammar_pairs(rng):
    pairs = []
    nouns = [(n, "en", "ett") for n in EN_NOUNS] + [(n, "ett", "en") for n in ETT_NOUNS]
    for i, (noun, good, bad) in enumerate(nouns):
        template = ARTICLE_TEMPLATES[i % len(ARTICLE_TEMPLATES)]
        pairs.append({"type": "article", "good": template.format(a=good, n=noun),
                      "bad": template.format(a=bad, n=noun)})
    for i, (noun, article, _) in enumerate(nouns):
        common, neuter = ADJECTIVES[i % len(ADJECTIVES)]
        good, bad = (common, neuter) if article == "en" else (neuter, common)
        pairs.append({"type": "adjective", "good": f"Vi har {article} {good} {noun}.",
                      "bad": f"Vi har {article} {bad} {noun}."})
    combos = set()
    while len(combos) < 80:
        combos.add((rng.choice(V2_ADVERBS), rng.choice(V2_SUBJECTS), rng.choice(V2_PREDICATES)))
    for adverb, subject, (verb, rest) in sorted(combos):
        pairs.append({"type": "word_order", "good": f"{adverb} {verb} {subject} {rest}.",
                      "bad": f"{adverb} {subject} {verb} {rest}."})
    return pairs


def multiple_choice(items, rng):
    out = []
    for question, correct, wrong in items:
        options = [correct] + wrong
        rng.shuffle(options)
        out.append({"question": question, "options": options, "answer": "ABCD"[options.index(correct)]})
    return out


def main():
    assert not set(DEV_PROMPTS) & set(TEST_PROMPTS)
    assert len(TEST_PROMPTS) == 100 and len(set(TEST_PROMPTS)) == 100
    rng = random.Random(0)
    files = {
        "src/prompts_dev.json": DEV_PROMPTS,
        "src/prompts_test.json": {"swedish": TEST_PROMPTS, "english": ENGLISH_PROMPTS},
        "src/probes.json": {"knowledge": multiple_choice(KNOWLEDGE, rng),
                            "general": multiple_choice(GENERAL, rng),
                            "grammar": grammar_pairs(rng)},
    }
    for path, content in files.items():
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(content, fh, ensure_ascii=False, indent=1)
        print("wrote", path)


if __name__ == "__main__":
    main()
