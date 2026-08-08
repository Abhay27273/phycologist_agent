"""10 diverse bilingual prompts — 20s spacing to avoid rate-limit fallbacks."""
import asyncio
import httpx

BASE = "http://127.0.0.1:8001"
DELAY = 20

PROMPTS = [
    ("EN", "dissociation",       "I feel completely numb. Like I am watching my life from outside."),
    ("HI", "hopeless mornings",  "main roz subah uthta hoon aur sochta hoon kya fayda sab karne ka"),
    ("EN", "panic at work",      "I keep having panic attacks at work. I had to hide in the bathroom today."),
    ("HI", "self-hate loop",     "mujhe apne aap se nafrat ho rahi hai. jo bhi karta hoon sab galat ho jaata hai"),
    ("EN", "crying no reason",   "I have been crying every day for two weeks and I do not even know why."),
    ("HI", "indirect crisis",    "khatam kar lena chahta hoon sab kuch. bahut thak gaya hoon is zindagi se"),
    ("EN", "empty achievement",  "I got the promotion I worked so hard for but I feel nothing. Empty."),
    ("HI", "caregiver pressure", "ghar pe maa ki bimari ki wajah se itna pressure hai ki saans lena mushkil ho gaya hai"),
    ("EN", "trust broken",       "I do not trust anyone anymore. Every time I open up I get hurt."),
    ("HI", "exam anxiety",       "kal se exam hai aur main khud ko concentrate hi nahi kar pa raha. sab bhool jaata hoon"),
]


async def main():
    async with httpx.AsyncClient(timeout=120) as c:
        r = await c.post(f"{BASE}/api/v1/auth/login",
                         json={"email": "run20_v2@example.com", "password": "test1234"})
        token = r.json()["access_token"]
        uid   = r.json()["user_id"]
        print(f"Auth OK  user={uid[:8]}...\n")
        headers = {"Authorization": f"Bearer {token}"}

        for i, (lang, label, text) in enumerate(PROMPTS, 1):
            sid = f"run10-s{i:02d}-{uid[:8]}"
            try:
                r2 = await c.post(f"{BASE}/api/v1/chat", headers=headers,
                                  json={"message": text, "user_id": uid, "session_id": sid})
                if r2.status_code == 200:
                    d    = r2.json()
                    mood = d.get("detected_mood", "?")
                    risk = d.get("risk_level", "?")
                    bot  = d.get("response", "")
                else:
                    mood, risk, bot = "ERROR", str(r2.status_code), r2.text[:200]
            except httpx.ReadTimeout:
                mood, risk, bot = "TIMEOUT", "—", "(request timed out after 120s)"
            except Exception as exc:
                mood, risk, bot = "EXCEPTION", "—", str(exc)[:200]

            print(f"[{i:02d}/10] {lang}  {label}  |  {mood} / {risk}")
            print(f"  YOU: {text}")
            print(f"  BOT: {bot}\n")

            if i < len(PROMPTS):
                await asyncio.sleep(DELAY)


if __name__ == "__main__":
    asyncio.run(main())
