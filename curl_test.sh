#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════════════════════
# Regime Risk Platform — curl Test Suite
# Tests POST /analyse/full with 200 synthetic multi-regime prices.
#
# Usage:
#   chmod +x curl_test.sh
#   ./curl_test.sh                        # runs against localhost:8000
#   ./curl_test.sh http://my-host:8000    # custom base URL
#
# Requirements: curl, python3 (for pretty-print), jq (optional, for colouring)
# ══════════════════════════════════════════════════════════════════════════════

BASE="${1:-http://localhost:8000}"
BOLD='\033[1m'; CYAN='\033[0;36m'; GREEN='\033[0;32m'; RED='\033[0;31m'; YELLOW='\033[0;33m'; RESET='\033[0m'

pass=0; fail=0

banner() { echo -e "\n${BOLD}${CYAN}── $1 ──${RESET}"; }
ok()     { echo -e "  ${GREEN}✓${RESET}  $1"; ((pass++)); }
err()    { echo -e "  ${RED}✗${RESET}  $1"; ((fail++)); }
info()   { echo -e "  ${YELLOW}→${RESET}  $1"; }

# ── Inline 200-bar payload (4-regime: low-vol bull → med bull → med bear → high bear) ──
PAYLOAD='{
  "symbol": "SPY",
  "prices": [
    100.3484,100.3793,100.8048,101.6732,101.6559,101.6385,102.5427,103.0387,
    102.8999,103.2819,103.1459,103.0089,103.2365,102.3521,101.5717,101.3877,
    100.9757,101.2353,100.8769,100.2655,101.1005,101.0875,101.2227,100.6028,
    100.4296,100.5857,100.1074,100.3956,100.1945,100.1486,99.9474,100.973,
    101.0672,100.6337,101.1482,100.632,100.8377,99.9505,99.3867,99.5839,
    100.0512,100.2369,100.2792,100.2285,99.5878,99.3289,99.1995,99.823,
    100.0944,99.3121,99.7281,99.2972,98.5204,99.2731,100.5311,101.6847,
    100.6912,100.3478,100.7768,101.9868,101.4309,101.2354,99.9217,98.5174,
    99.5075,101.1569,101.0998,102.3476,102.8225,102.0573,102.5305,104.4536,
    104.4401,106.4323,103.1184,104.1663,104.3064,103.9634,104.1091,101.6572,
    101.4198,101.8848,103.7223,103.1083,102.1389,101.5545,102.7006,103.1365,
    102.5118,103.174,103.3251,104.5571,103.7076,103.331,102.8758,101.0999,
    101.4895,101.8379,101.8747,101.6185,98.6606,97.7517,97.0034,95.3694,
    94.9854,95.677,99.2098,99.4768,99.9096,99.6809,95.7759,95.6485,
    95.6872,100.3247,99.8584,100.3808,100.2308,97.8079,99.9651,101.3885,
    102.9114,100.9574,103.7091,100.7184,101.82,106.1992,104.0103,102.7491,
    102.8717,101.7535,98.5164,98.5727,96.3995,97.2355,95.3697,98.2497,
    96.6321,95.9323,97.4164,94.9404,95.2963,97.7114,94.4918,94.7652,
    95.1819,96.5941,94.1272,91.566,92.4486,92.9238,93.4548,94.2579,
    92.1653,92.6416,93.2894,91.1222,96.0586,97.2512,93.6004,95.2756,
    92.3182,94.3319,97.4408,94.8664,97.4374,98.4686,100.7198,106.2698,
    105.2962,102.7257,99.7996,97.1774,96.7777,97.594,98.2284,100.4892,
    100.3475,104.5426,103.5244,111.7862,113.6832,110.5552,106.8045,108.1581,
    107.2383,109.3424,110.6979,110.2568,107.2574,102.19,100.6371,103.0415,
    103.5179,99.4629,99.8006,100.7746,97.9211,98.1964,98.1911,94.6475
  ],
  "kelly_fraction": 0.5,
  "target_vol": 0.15,
  "base_risk_limit": 0.02
}'

# ─── 1. Health check ──────────────────────────────────────────────────────────
banner "1. Health check"
HTTP=$(curl -s -o /dev/null -w "%{http_code}" "${BASE}/health")
BODY=$(curl -s "${BASE}/health")
if [[ "$HTTP" == "200" ]]; then
  ok "GET /health → 200"
  info "$(echo $BODY | python3 -m json.tool 2>/dev/null || echo $BODY)"
else
  err "GET /health → $HTTP (is the server running? uvicorn main:app --port 8000)"
  exit 1
fi

# ─── 2. Full pipeline ─────────────────────────────────────────────────────────
banner "2. POST /analyse/full — 200-bar SPY, 4 multi-regime segments"
info "Sending 200 prices (bars 1-50: low-vol bull, 51-100: med bull, 101-150: med bear, 151-200: high-vol bear)"

RESPONSE=$(curl -s -w "\n%{http_code}" \
  -X POST "${BASE}/analyse/full" \
  -H "Content-Type: application/json" \
  -d "$PAYLOAD")

HTTP_CODE=$(echo "$RESPONSE" | tail -1)
BODY=$(echo "$RESPONSE" | head -n -1)

if [[ "$HTTP_CODE" == "200" ]]; then
  ok "POST /analyse/full → 200"
else
  err "POST /analyse/full → $HTTP_CODE"
  echo "$BODY" | python3 -m json.tool 2>/dev/null || echo "$BODY"
  exit 1
fi

# ─── 3. Assert response fields ────────────────────────────────────────────────
banner "3. Response field assertions"

check_field() {
  local label="$1"; local value="$2"; local expected="$3"
  if [[ -n "$value" && "$value" != "null" ]]; then
    ok "$label = $value"
  else
    err "$label missing or null (expected $expected)"
  fi
}

REGIME_LABEL=$(echo "$BODY"  | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['regime']['regime_label'])" 2>/dev/null)
VOL_STATE=$(echo "$BODY"     | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['regime']['vol_state'])" 2>/dev/null)
TREND_STATE=$(echo "$BODY"   | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['regime']['trend_state'])" 2>/dev/null)
CONF=$(echo "$BODY"          | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['regime']['confidence'])" 2>/dev/null)
RISK_MULT=$(echo "$BODY"     | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['regime']['risk_multiplier'])" 2>/dev/null)
REC_F=$(echo "$BODY"         | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['sizing']['recommended_f'])" 2>/dev/null)
SHARPE=$(echo "$BODY"        | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['sizing']['sharpe_ratio'])" 2>/dev/null)
VAR95=$(echo "$BODY"         | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['risk']['var_95'])" 2>/dev/null)
CVAR95=$(echo "$BODY"        | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['risk']['cvar_95'])" 2>/dev/null)
MAX_DD=$(echo "$BODY"        | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['risk']['max_drawdown'])" 2>/dev/null)
STOP=$(echo "$BODY"          | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['risk']['suggested_stop'])" 2>/dev/null)
TP=$(echo "$BODY"            | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['risk']['suggested_tp'])" 2>/dev/null)

check_field "regime_label"     "$REGIME_LABEL"  "e.g. high_vol_strong_bear"
check_field "vol_state"        "$VOL_STATE"     "low|med_low|med_high|high"
check_field "trend_state"      "$TREND_STATE"   "strong_bear|bear|bull|strong_bull"
check_field "confidence"       "$CONF"          "0–1"
check_field "risk_multiplier"  "$RISK_MULT"     "0.10–1.75"
check_field "recommended_f"    "$REC_F"         "0–1"
check_field "sharpe_ratio"     "$SHARPE"        "float"
check_field "var_95"           "$VAR95"         ">0"
check_field "cvar_95"          "$CVAR95"        ">0"
check_field "max_drawdown"     "$MAX_DD"        "<0"
check_field "suggested_stop"   "$STOP"          "<0"
check_field "suggested_tp"     "$TP"            ">0"

# ─── 4. Pretty-print result ───────────────────────────────────────────────────
banner "4. Full response"
if command -v jq &>/dev/null; then
  echo "$BODY" | jq '{
    regime: .regime | {regime_label, vol_state, trend_state, confidence, risk_multiplier},
    sizing: .sizing | {recommended_f, sharpe_ratio, fraction_type, notes},
    risk:   .risk   | {var_95, cvar_95, max_drawdown, sortino_ratio, suggested_stop, suggested_tp}
  }'
else
  echo "$BODY" | python3 -m json.tool
fi

# ─── 5. Edge cases ────────────────────────────────────────────────────────────
banner "5. Edge case — fewer than 81 bars (expect 422)"
SHORT_PAYLOAD='{"symbol":"SPY","prices":[100,101,102,103,104,105],"kelly_fraction":0.5,"target_vol":0.15,"base_risk_limit":0.02}'
HTTP_SHORT=$(curl -s -o /dev/null -w "%{http_code}" \
  -X POST "${BASE}/analyse/full" \
  -H "Content-Type: application/json" \
  -d "$SHORT_PAYLOAD")
if [[ "$HTTP_SHORT" == "422" ]]; then
  ok "6-bar payload correctly rejected → 422"
else
  err "Expected 422, got $HTTP_SHORT"
fi

banner "6. Edge case — negative price (expect 422)"
BAD_PAYLOAD='{"symbol":"SPY","prices":[-10,100,101],"kelly_fraction":0.5,"target_vol":0.15,"base_risk_limit":0.02}'
HTTP_BAD=$(curl -s -o /dev/null -w "%{http_code}" \
  -X POST "${BASE}/analyse/full" \
  -H "Content-Type: application/json" \
  -d "$BAD_PAYLOAD")
if [[ "$HTTP_BAD" == "422" ]]; then
  ok "Negative price correctly rejected → 422"
else
  err "Expected 422, got $HTTP_BAD"
fi

# ─── Summary ──────────────────────────────────────────────────────────────────
echo ""
echo "══════════════════════════════════════════════"
total=$((pass + fail))
if [[ $fail -eq 0 ]]; then
  echo -e "  ${GREEN}${BOLD}✓ $pass/$total passed${RESET}"
else
  echo -e "  ${RED}${BOLD}✗ $fail failed  ✓ $pass passed${RESET}"
fi
echo "══════════════════════════════════════════════"
echo ""
[[ $fail -eq 0 ]]   # exit 0 on success, 1 on failure
