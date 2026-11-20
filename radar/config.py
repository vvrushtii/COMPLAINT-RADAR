"""Shared settings. Edit APPS to track more companies; edit THEMES to change what we look for."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"

# display name -> Google Play app id
APPS = {
    "Swiggy": "in.swiggy.android",
    "Zomato": "com.application.zomato",
    "Blinkit": "com.grofers.customerapp",
    "Zepto": "com.zeptoconsumerapp",
}

EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"   # handles Hindi/Hinglish too

# theme -> (plain-English description used by the zero-shot model, keyword regex used by the rule baseline)
THEMES = {
    "Late / slow delivery":     ("order arrived very late, long delay, slow delivery, waiting for hours",
                                 r"\blate\b|delay|\bslow\b|too long|waiting|waited|hour|never arrive|not deliver|not arrive"),
    "Order cancelled by app":   ("the app or restaurant cancelled my order",
                                 r"cancel"),
    "Refund / money issues":    ("refund not received, money deducted, payment problem, charged twice",
                                 r"refund|money|charged|deduct|payment|wallet|double charge"),
    "Customer support":         ("customer support is useless, no response from customer care, chatbot only",
                                 r"support|customer care|customer service|no response|chat bot|chatbot|helpline|agent"),
    "Wrong / missing items":    ("wrong item delivered, item missing from the order, spilled or damaged package",
                                 r"wrong|missing|missed|incorrect|spilled|spill|damaged"),
    "Food / product quality":   ("food was cold, stale, bad quality, or groceries were rotten or expired",
                                 r"cold food|stale|quality|taste|tasteless|hygien|raw|smell|spoil|rotten|expired"),
    "High fees / pricing":      ("too many fees, platform fee, delivery charge, prices are expensive",
                                 r"fee|charge|expensive|costly|price|surge|overpric|handling"),
    "App bugs / login / crash": ("app crashes, login problem, OTP not coming, app not working or stuck",
                                 r"\bbug|crash|login|log in|otp|hang|glitch|not working|stuck|freez|error|loading"),
    "Delivery partner conduct": ("delivery partner was rude, misbehaved, or did not follow instructions",
                                 r"delivery partner|delivery boy|rider|rude|behav|misbehav"),
    "Offers / membership":      ("coupon or offer did not work, membership benefits missing, cashback issue",
                                 r"coupon|offer|discount|promo|swiggy one|membership|subscription|cashback|zomato gold|gold"),
}
THEME_NAMES = list(THEMES)
