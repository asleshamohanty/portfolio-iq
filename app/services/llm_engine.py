"""
app/services/llm_engine.py

Sprint 5 — Financial Reasoning Layer.

The LLM's job is synthesis, not calculation.
Every number in the response comes from the quant engine, ML engine,
or RAG retrieval. The LLM's only job is to explain what those numbers
mean in plain English, grounded in retrieved evidence.

Flow for /chat:
  1. Pull live risk metrics from quant engine
  2. Pull ML volatility forecast
  3. Retrieve relevant document chunks via RAG
  4. Pass all three as structured context to the LLM
  5. Return grounded explanation with metric references

Flow for /chat/stress-test:
  1. Parse the plain-English scenario
  2. Map to historical analogue period via RAG + keyword matching
  3. Run quant engine stress test on that period
  4. Pass results to LLM for synthesis
  5. Return grounded scenario analysis
"""

import os
from google import genai
from google.genai import types
from sqlalchemy.orm import Session

from app.services.quant_engine import full_risk_report, stress_test
from app.services.ml_engine import train_and_predict
from app.services.rag_engine import search_documents

client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
MODEL = "gemini-2.5-flash"
# ── Prompt templates ──────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a senior portfolio risk analyst writing institutional-grade research reports for portfolio managers at firms like Goldman Sachs and BlackRock.

Your responses must read like a professional risk report — flowing prose, no bullet points, no numbered lists, no markdown headers. Write in clear, confident paragraphs the way a Managing Director would brief a client.

Critical rules:
- Never use bullet points, numbered lists, or markdown formatting of any kind.
- Write in complete paragraphs with natural transitions between ideas.
- Every number you cite must come from the context provided — never invent figures.
- Reference your sources naturally in prose: "According to Goldman Sachs's 2025 10-K...", "The ML model projects...", "Historical analysis of the 2022 rate cycle shows..."
- Express volatility and returns as percentages, not decimals.
- Lead with the key insight in the first sentence. Support it with evidence. Close with a clear recommendation.
- Be direct. Portfolio managers are busy. One paragraph per idea, maximum four paragraphs total.
"""

CHAT_PROMPT_TEMPLATE = """
## Portfolio Context

**Holdings and weights:**
{weights}

**Quantitative Risk Metrics (last {lookback_days} trading days):**
- Annualised volatility: {volatility}%
- Sharpe ratio: {sharpe}
- Sortino ratio: {sortino}
- Maximum drawdown: {max_drawdown}%
- Value at Risk (95%): {var_95}% per day
- CVaR (95%): {cvar_95}% per day

**Sector Exposure:**
{sector_exposure}

**ML Forecast:**
- Predicted 30-day volatility: {predicted_vol}%
- Current 21-day volatility: {current_vol}%
- Signal: {signal}
- Top drivers: {shap_features}

**Stress Test Results:**
{stress_tests}

**Retrieved Document Context:**
{rag_context}

---

## User Question
{question}

Answer based strictly on the context above. Be specific, reference the numbers, 
and cite document sources where relevant.
"""

STRESS_TEST_PROMPT_TEMPLATE = """
## Stress Scenario Analysis

**Portfolio:**
{weights}

**Scenario:** "{scenario}"

**Historical Analogue Results:**
{stress_results}

**Current Portfolio Risk Profile:**
- Annualised volatility: {volatility}%  
- Sharpe ratio: {sharpe}
- Maximum drawdown: {max_drawdown}%
- Sector exposure: {sector_exposure}

**Retrieved Document Context (relevant to this scenario):**
{rag_context}

---

Analyse this stress scenario for the portfolio above.

Write a four-paragraph institutional risk report on this scenario. First paragraph: what this scenario means historically and how it has played out. Second paragraph: how this specific portfolio would be affected, citing the actual stress test numbers. Third paragraph: which holdings and sector exposures create the most vulnerability, referencing what the retrieved filings say about this type of risk. Fourth paragraph: two or three specific, actionable portfolio adjustments with clear rationale. Write in flowing prose, no lists, no headers, no markdown.

Be specific. Reference the actual numbers from the stress test results.
"""


# ── Helper formatters ─────────────────────────────────────────────────────────

def _format_weights(weights: dict) -> str:
    return "\n".join(f"  - {sym}: {w*100:.1f}%" for sym, w in weights.items())


def _format_sector_exposure(exposure: dict) -> str:
    return "\n".join(f"  - {sector}: {w*100:.1f}%" for sector, w in exposure.items())


def _format_shap(features: list) -> str:
    return ", ".join(
        f"{f['feature']} ({f['direction'].split()[0]})"
        for f in features[:3]
    )


def _format_stress_tests(stress: dict) -> str:
    lines = []
    for scenario, data in stress.items():
        if "error" in data:
            lines.append(f"  - {scenario}: insufficient historical data")
        else:
            lines.append(
                f"  - {scenario}: cumulative return {data['cumulative_return']*100:.1f}%, "
                f"max drawdown {data['max_drawdown']*100:.1f}%, "
                f"volatility {data['volatility']*100:.1f}%"
            )
    return "\n".join(lines)


def _format_rag_context(chunks: list) -> str:
    if not chunks:
        return "No relevant documents retrieved for this query."
    lines = []
    for i, chunk in enumerate(chunks, 1):
        lines.append(
            f"[Source {i}: {chunk['symbol']} {chunk['doc_type']} "
            f"({chunk['period']}) — similarity {chunk['similarity']}]\n"
            f"{chunk['chunk_text'][:600]}..."
        )
    return "\n\n".join(lines)


def _format_stress_results(stress: dict) -> str:
    lines = []
    for scenario, data in stress.items():
        if "error" not in data:
            lines.append(
                f"**{scenario.replace('_', ' ').title()}** "
                f"({data['period']}):\n"
                f"  - Cumulative return: {data['cumulative_return']*100:.1f}%\n"
                f"  - Max drawdown: {data['max_drawdown']*100:.1f}%\n"
                f"  - Annualised volatility: {data['volatility']*100:.1f}%"
            )
    return "\n\n".join(lines) if lines else "No historical stress data available for comparison."


# ── Main functions ────────────────────────────────────────────────────────────

def chat(
    db: Session,
    question: str,
    weights: dict[str, float],
    lookback_days: int = 252,
) -> dict:
    """
    Answer a natural language question about a portfolio.

    Pipeline:
      quant metrics + ML forecast + RAG retrieval → LLM synthesis
    """
    # 1. Quant engine
    risk = full_risk_report(db, weights, lookback_days)
    if "error" in risk:
        return {"error": risk["error"]}

    metrics = risk["metrics"]

    # 2. ML forecast
    ml = train_and_predict(db, weights)
    predicted_vol = ml.get("predicted_30d_volatility", "unavailable")
    current_vol   = ml.get("current_21d_volatility", "unavailable")
    signal        = ml.get("signal", "unavailable")
    shap_features = ml.get("shap_top5_features", [])

    # 3. RAG — search using the question + portfolio symbols
    rag_query  = f"{question} {' '.join(weights.keys())}"
    rag_chunks = search_documents(db, rag_query, top_k=3)

    # 4. Build structured prompt
    prompt = CHAT_PROMPT_TEMPLATE.format(
        weights       = _format_weights(weights),
        lookback_days = lookback_days,
        volatility    = f"{metrics['annualised_volatility']*100:.2f}",
        sharpe        = f"{metrics['sharpe_ratio']:.3f}",
        sortino       = f"{metrics['sortino_ratio']:.3f}",
        max_drawdown  = f"{metrics['max_drawdown']*100:.2f}",
        var_95        = f"{metrics['var_95']*100:.2f}",
        cvar_95       = f"{metrics['cvar_95']*100:.2f}",
        sector_exposure = _format_sector_exposure(risk.get("sector_exposure", {})),
        predicted_vol = f"{predicted_vol*100:.2f}" if isinstance(predicted_vol, float) else predicted_vol,
        current_vol   = f"{current_vol*100:.2f}" if isinstance(current_vol, float) else current_vol,
        signal        = signal,
        shap_features = _format_shap(shap_features),
        stress_tests  = _format_stress_tests(risk.get("stress_tests", {})),
        rag_context   = _format_rag_context(rag_chunks),
        question      = question,
    )

    # 5. LLM synthesis
    response = client.models.generate_content(
        model=MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            max_output_tokens=1024,
        ),
    )
    answer = response.text

    return {
        "question": question,
        "answer":   answer,
        "sources":  [
            {"symbol": c["symbol"], "doc_type": c["doc_type"],
             "period": c["period"], "similarity": c["similarity"]}
            for c in rag_chunks
        ],
        "metrics_used": {
            "volatility":   metrics["annualised_volatility"],
            "sharpe":       metrics["sharpe_ratio"],
            "predicted_vol": predicted_vol,
        },
    }


def stress_test_chat(
    db: Session,
    scenario: str,
    weights: dict[str, float],
) -> dict:
    """
    The flagship endpoint.

    Takes a plain-English scenario like "Fed raises rates 200bps",
    runs the portfolio through historical stress periods, retrieves
    relevant document context, and returns a grounded analysis.
    """
    # 1. Current risk profile
    risk = full_risk_report(db, weights)
    if "error" in risk:
        return {"error": risk["error"]}

    metrics = risk["metrics"]

    # 2. Stress test results
    stress_results = risk.get("stress_tests", {})

    # 3. RAG — retrieve docs relevant to the scenario
    rag_chunks = search_documents(db, scenario, top_k=4)

    # Also search with portfolio-specific context
    portfolio_query = f"{scenario} portfolio risk {' '.join(weights.keys())}"
    extra_chunks    = search_documents(db, portfolio_query, top_k=2)

    # Deduplicate chunks by chunk_text
    seen    = set()
    all_chunks = []
    for chunk in rag_chunks + extra_chunks:
        key = chunk["chunk_text"][:100]
        if key not in seen:
            seen.add(key)
            all_chunks.append(chunk)

    # 4. Build structured prompt
    prompt = STRESS_TEST_PROMPT_TEMPLATE.format(
        weights        = _format_weights(weights),
        scenario       = scenario,
        stress_results = _format_stress_results(stress_results),
        volatility     = f"{metrics['annualised_volatility']*100:.2f}",
        sharpe         = f"{metrics['sharpe_ratio']:.3f}",
        max_drawdown   = f"{metrics['max_drawdown']*100:.2f}",
        sector_exposure = _format_sector_exposure(risk.get("sector_exposure", {})),
        rag_context    = _format_rag_context(all_chunks),
    )

    # 5. LLM synthesis
    response = client.models.generate_content(
        model=MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            max_output_tokens=1500,
        ),
    )
    answer = response.text

    return {
        "scenario":      scenario,
        "analysis":      answer,
        "stress_data":   stress_results,
        "sources_used":  len(all_chunks),
        "sources": [
            {"symbol": c["symbol"], "doc_type": c["doc_type"],
             "period": c["period"], "similarity": c["similarity"]}
            for c in all_chunks
        ],
    }