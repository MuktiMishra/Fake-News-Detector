"""
generate_dataset.py
Generates a sample CSV dataset for demonstration/testing purposes.
For best results, replace with real Kaggle Fake News Dataset.
"""

import pandas as pd
import random

random.seed(42)

fake_headlines = [
    "SHOCKING: Government secretly puts chemicals in water supply to control population",
    "Scientists CONFIRM vaccines cause autism in new study that mainstream media won't show you",
    "Breaking: President arrested for treason, military takeover imminent",
    "LEAKED: CIA documents prove moon landing was filmed in Hollywood",
    "Miracle cure for cancer discovered but pharmaceutical companies are hiding it",
    "Aliens working with government confirmed by whistleblower",
    "Bill Gates admits microchips inserted into COVID vaccines in secret recording",
    "Earth is actually flat, NASA whistleblower reveals the truth",
    "Democrats planning to ban all guns by end of month, secret memo leaked",
    "5G towers proven to spread coronavirus, study reveals",
    "Global elite planning to reduce world population by 90 percent",
    "New world order meeting held in secret underground bunker confirmed",
    "Hollywood celebrity arrested for running child trafficking ring",
    "Banks planning to seize all personal accounts next week",
    "Ancient ruins found on Mars confirm civilization existed",
    "Doctor reveals drinking bleach can cure COVID-19",
    "Tech company admits to mind-reading technology installed in smartphones",
    "Secret treaty hands US sovereignty to United Nations",
    "Mainstream media admits to fabricating all news stories since 2010",
    "Major earthquake to hit California next week, government hiding the prediction",
]

fake_bodies = [
    "Sources close to the government have revealed shocking information that mainstream media refuses to cover. Multiple whistleblowers have come forward with evidence that cannot be ignored by anyone who is paying attention to what is really going on in our world today.",
    "A leaked document obtained by independent researchers shows what the deep state does not want you to know. Share this before it gets deleted. The truth is being suppressed by powerful interests who control what you see and hear.",
    "An anonymous insider with top-level security clearance has gone on record to expose the lies we have been told our entire lives. The evidence is overwhelming and irrefutable. Do your own research and wake up to what is happening.",
    "Exclusive sources from inside the organization have confirmed what conspiracy theorists have been saying for years. The mainstream media will not report this because they are controlled by the same people who are behind this cover-up.",
    "New documents released by a courageous whistleblower reveal the shocking extent of the deception. Millions of Americans have been kept in the dark about this for decades. The truth can no longer be hidden from the public.",
]

real_headlines = [
    "Federal Reserve raises interest rates by 0.25 percent amid inflation concerns",
    "Scientists discover new species of deep-sea fish in Pacific Ocean",
    "City council approves new budget for road infrastructure repairs",
    "Study finds Mediterranean diet linked to improved heart health outcomes",
    "Tech company announces quarterly earnings above analyst expectations",
    "Local hospital opens new cancer treatment center after fundraising campaign",
    "Weather service issues warning for potential winter storm next week",
    "University researchers develop more efficient solar panel technology",
    "Sports team advances to championship after overtime victory",
    "New legislation aims to reduce carbon emissions from industrial sector",
    "International summit on climate change concludes with updated commitments",
    "Central bank holds interest rates steady, cites economic uncertainty",
    "Archaeologists uncover ancient settlement during highway construction project",
    "Public health officials report decline in flu cases this season",
    "Company recalls product due to potential safety hazard, consumers urged to return",
    "Election results certified after routine audit confirms accuracy",
    "New report highlights progress in renewable energy adoption worldwide",
    "City announces plan to expand public transportation network by 2026",
    "Researchers publish findings on effectiveness of new diabetes medication",
    "Annual report shows improvement in national literacy rates among children",
]

real_bodies = [
    "According to official statements released on Tuesday, the decision was made following extensive review by a committee of experts. The move is expected to have significant implications for the sector in the coming months.",
    "In a press conference held at the organization headquarters, officials outlined the key details of the announcement. Experts in the field have offered mixed reactions, with some calling it a positive development and others raising concerns.",
    "The findings, published in a peer-reviewed journal, represent months of careful research by a team of specialists. The study involved participants from multiple regions and followed established scientific protocols.",
    "Government officials confirmed the policy change in an official statement, noting that the decision was reached after consultation with relevant stakeholders. Implementation is expected to begin in the next fiscal quarter.",
    "The report, compiled from data gathered over the past two years, presents a comprehensive analysis of trends in the industry. Analysts say the results align with broader economic patterns observed in recent quarters.",
]

data = []

for i in range(500):
    headline = random.choice(fake_headlines)
    body = random.choice(fake_bodies)
    data.append({
        "title": headline,
        "text": body + " " + body,
        "label": 1  # 1 = FAKE
    })

for i in range(500):
    headline = random.choice(real_headlines)
    body = random.choice(real_bodies)
    data.append({
        "title": headline,
        "text": body + " " + body,
        "label": 0  # 0 = REAL
    })

df = pd.DataFrame(data)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
df.to_csv("dataset/news.csv", index=False)
print(f"Dataset created: {len(df)} rows saved to dataset/news.csv")
print(df["label"].value_counts())
