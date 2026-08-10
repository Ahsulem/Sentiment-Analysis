from dataclasses import dataclass


@dataclass
class SentimentResult:
    label: str
    score: float


class YouTubeSentimentModel:
    """A lightweight sentiment model for YouTube comment text."""

    POSITIVE_WORDS = {
        "good",
        "great",
        "love",
        "awesome",
        "excellent",
        "amazing",
        "best",
        "nice",
        "happy",
    }
    NEGATIVE_WORDS = {
        "bad",
        "worst",
        "hate",
        "awful",
        "terrible",
        "poor",
        "sad",
        "boring",
        "disappointing",
    }

    def predict(self, text: str) -> SentimentResult:
        tokens = [token.strip(".,!?;:").lower() for token in text.split() if token.strip()]
        if not tokens:
            return SentimentResult(label="neutral", score=0.0)

        positive = sum(token in self.POSITIVE_WORDS for token in tokens)
        negative = sum(token in self.NEGATIVE_WORDS for token in tokens)
        score = (positive - negative) / len(tokens)

        if score > 0:
            label = "positive"
        elif score < 0:
            label = "negative"
        else:
            label = "neutral"

        return SentimentResult(label=label, score=round(score, 4))
