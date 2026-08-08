"""20 new bilingual prompts — records per-request latency, prints p50/p95/p99."""
import asyncio
import sys
import time
import httpx

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE  = "http://127.0.0.1:8001"
DELAY = 20

PROMPTS = [
    ("EN", "grief anniversary",   "It is my mom's death anniversary today and I do not know how to get through the day."),
    ("HI", "workplace bias",      "office mein mujhse sirf isliye bura behave kiya jaata hai kyunki main naya hoon. koi nahi sunta"),
    ("EN", "invisible in rooms",  "I talk and people just look past me. It is like I do not exist in rooms."),
    ("HI", "debt secret",         "teen lakh ka karz ho gaya hai. kisi ko pata nahi. roz uthke ek natak karta hoon ki sab theek hai"),
    ("EN", "family estrangement", "I cut off my parents two years ago for my mental health. I still feel guilty every single day."),
    ("HI", "peer comparison",     "mere dost sab settle ho gaye hain. main abhi bhi struggle kar raha hoon. bahut inferior feel hota hai"),
    ("EN", "unforgivable words",  "I said something unforgivable to my best friend. They have not spoken to me in three months."),
    ("HI", "empty nest",          "bachche university gaye. ghar bahut suna lagta hai. pata nahi main kaun hoon ab"),
    ("EN", "never checked on",    "Growing up nobody ever checked on me. I raised myself. I am still waiting for someone to ask if I am okay."),
    ("HI", "spiritual crisis",    "pehle bhagwan pe bharosa tha. ek ghatna ne sab tod diya. ab pooja karna bhi bekaar lagta hai"),
    ("EN", "fear abandonment",    "Every time someone gets close to me I start pushing them away. I can see myself doing it and cannot stop."),
    ("HI", "domestic fear",       "ghar mein bahut darna lagta hai. kabhi kabhi darr ke room mein band ho jaata hoon"),
    ("EN", "retirement hollow",   "I retired six months ago. I thought I would be happy. Instead I feel useless every day."),
    ("HI", "chronic lonely",      "saalon se koi dost nahi hai. logo se baat karna nahi aata. akela rehna hi theek lagta hai ab"),
    ("EN", "failed semester",     "I failed my semester. My parents sacrificed so much for me. I cannot face them."),
    ("HI", "diaspora homesick",   "videsh mein hoon. yahaan koi apna nahi. log bahut alag hain. ghar ki yaad mein aankhein bhar aati hain"),
    ("EN", "caregiver 3yrs",      "I have been taking care of my husband with dementia for three years. I have not had a single day for myself."),
    ("HI", "relapse shame",       "6 mahine sober tha. phir kal pee liya. khud se itni nafrat ho rahi hai"),
    ("EN", "guilty laugh",        "I actually laughed today for the first time in months. A real laugh. And then I felt guilty about it."),
    ("HI", "night eating",        "raat ko uthke khaana khata hoon. subah yaad bhi nahi rehta. thak gaya hoon khud se"),
]


def percentile(data: list[float], p: float) -> float:
    if not data:
        return 0.0
    s = sorted(data)
    idx = (p / 100) * (len(s) - 1)
    lo, hi = int(idx), min(int(idx) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (idx - lo)


async def main():
    latencies: list[float] = []

    async with httpx.AsyncClient(timeout=120) as client:
        r = await client.post(f"{BASE}/api/v1/auth/register",
                              json={"email": "run20_v5@example.com", "password": "test1234"})
        if r.status_code not in (200, 201) and "already registered" not in r.text:
            print(f"Register failed: {r.text}"); return
        if r.status_code not in (200, 201):
            r = await client.post(f"{BASE}/api/v1/auth/login",
                                  json={"email": "run20_v5@example.com", "password": "test1234"})
        token   = r.json()["access_token"]
        user_id = r.json().get("user_id", "run20v5")
        print(f"Auth OK  user={user_id[:8]}...\n")
        headers = {"Authorization": f"Bearer {token}"}

        for i, (lang, label, text) in enumerate(PROMPTS, 1):
            session_id = f"run20v5-s{i:02d}-{user_id[:8]}"
            t0 = time.perf_counter()
            try:
                r2 = await client.post(f"{BASE}/api/v1/chat", headers=headers,
                                       json={"message": text, "user_id": user_id, "session_id": session_id})
                elapsed = time.perf_counter() - t0
                if r2.status_code == 200:
                    d    = r2.json()
                    mood = d.get("detected_mood", "?")
                    risk = d.get("risk_level", "?")
                    bot  = d.get("response", "")
                    latencies.append(elapsed)
                else:
                    mood, risk, bot = "ERROR", str(r2.status_code), r2.text[:200]
                    elapsed = time.perf_counter() - t0
            except httpx.ReadTimeout:
                elapsed = time.perf_counter() - t0
                mood, risk, bot = "TIMEOUT", "—", "(timed out after 120s)"
            except Exception as exc:
                elapsed = time.perf_counter() - t0
                mood, risk, bot = "EXCEPTION", "—", str(exc)[:200]

            print(f"[{i:02d}/20] {lang}  {label}  |  {mood} / {risk}  [{elapsed:.2f}s]")
            print(f"  YOU: {text}")
            print(f"  BOT: {bot}\n")

            if i < len(PROMPTS):
                await asyncio.sleep(DELAY)

    if latencies:
        print("─" * 60)
        print(f"  Requests completed : {len(latencies)}/20")
        print(f"  Min                : {min(latencies):.2f}s")
        print(f"  p50 (median)       : {percentile(latencies, 50):.2f}s")
        print(f"  p95                : {percentile(latencies, 95):.2f}s")
        print(f"  p99                : {percentile(latencies, 99):.2f}s")
        print(f"  Max                : {max(latencies):.2f}s")
        print("─" * 60)


if __name__ == "__main__":
    asyncio.run(main())
