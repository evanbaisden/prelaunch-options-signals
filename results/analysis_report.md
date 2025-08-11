# Prelaunch Options Signals - Phase 1 Analysis Report

## Executive Summary

This analysis examines stock price and volume patterns around major technology product launches to identify potential trading signals and information leakage. The study covers eight significant product launches:

1. **Microsoft Xbox Series X/S** (Announcement: Sep 9, 2020 | Release: Nov 10, 2020)
2. **NVIDIA RTX 30 Series** (Announcement: Sep 1, 2020 | Release: Sep 17, 2020)  
3. **NVIDIA RTX 40 Series** (Announcement: Sep 20, 2022 | Release: Oct 12, 2022)
4. **NVIDIA RTX 40 SUPER** (Announcement: Jan 8, 2024 | Release: Jan 17, 2024)
5. **Apple iPhone 12** (Announcement: Oct 13, 2020 | Release: Oct 23, 2020)
6. **Apple iPhone 13** (Announcement: Sep 14, 2021 | Release: Sep 24, 2021)
7. **Apple iPhone 14** (Announcement: Sep 7, 2022 | Release: Sep 16, 2022)
8. **Apple iPhone 15** (Announcement: Sep 12, 2023 | Release: Sep 22, 2023)

## Methodology and Calculation Definitions

**Critical Note**: All calculations use precise definitions to ensure reproducibility:

### Return Calculations
- **5-Day Announcement Return**: Cumulative return from `close[t-5]` to `close[t+0]` on announcement day
- **5-Day Release Return**: Cumulative return from `close[t-5]` to `close[t+0]` on release day  
- **Pre-Announcement Return**: Average daily return over 60 trading days before announcement
- **Announcement-to-Release Return**: Average daily return from announcement day to release day
- **Post-Release Return**: Average daily return over 30 trading days after release

*Note: All price returns use adjusted close prices (yfinance) to account for stock splits and dividends. Volume data represents raw daily trading volumes without adjustments.*

### Volume Calculations
- **Volume Spike %**: `(volume[period] / baseline_mean) - 1`, where baseline = 20-day moving average
- **Baseline Volume**: 20-day rolling mean with `min_periods=10` for early data handling
- **Volume Anomaly Detection**: Z-score method with thresholds at 1.645 (95th percentile, one-tailed), 2.326 (99th percentile, one-tailed), 2.576 (99% two-tailed / 99.5% one-tailed), and 5.0 (extreme outliers)

### Analysis Parameters (Configurable via .env)
- **Baseline Period**: 60 trading days for trend analysis
- **Signal Windows**: 5 days before/20 days after announcements; 5 days before/10 days after releases
- **Statistical Thresholds**: Z-scores for anomaly detection at multiple confidence levels

## Key Findings

### 1. Price Movement Patterns

**Microsoft Xbox Series X/S:**
- Pre-announcement daily return: **+0.15%** (positive momentum building)
- Announcement to release: **+0.19%** (continued positive sentiment)
- Post-release: **-0.15%** (sell-the-news effect)
- 5-day announcement return: **-7.03%** (negative surprise)
- 5-day release return: **+2.22%** (recovery)

**NVIDIA RTX 30 Series:**
- Pre-announcement daily return: **+0.70%** (strong bullish momentum)
- Announcement to release: **-0.48%** (profit-taking period)
- Post-release: **+0.06%** (stabilization)
- 5-day announcement return: **+8.43%** (strong positive reaction)
- 5-day release return: **+1.23%** (continued strength)

**NVIDIA RTX 40 Series:**
- Pre-announcement daily return: **-0.25%** (bearish environment)
- Announcement to release: **-0.84%** (continued weakness)
- Post-release: **+1.17%** (strong recovery)
- 5-day announcement return: **+0.34%** (muted reaction)
- 5-day release return: **-12.94%** (significant sell-off)

**NVIDIA RTX 40 SUPER:**
- Pre-announcement daily return: **+0.14%** (neutral trend)
- Announcement to release: **+2.35%** (strong positive momentum)
- Post-release: **+1.13%** (sustained gains)
- 5-day announcement return: **+5.51%** (positive market reaction)
- 5-day release return: **+5.48%** (continued strength)

**Apple iPhone 12:**
- Pre-announcement daily return: **+0.48%** (strong momentum building)
- Announcement to release: **-0.89%** (profit-taking period)
- Post-release: **+0.06%** (stabilization)
- 5-day announcement return: **+7.02%** (strong positive reaction)
- 5-day release return: **-3.34%** (sell-the-news effect)

**Apple iPhone 13:**
- Pre-announcement daily return: **+0.22%** (moderate momentum)
- Announcement to release: **-0.22%** (neutral sentiment)
- Post-release: **+0.10%** (slight recovery)
- 5-day announcement return: **-5.47%** (negative surprise)
- 5-day release return: **+0.59%** (minor recovery)

**Apple iPhone 14:**
- Pre-announcement daily return: **+0.15%** (weak momentum)
- Announcement to release: **-0.16%** (slightly bearish)
- Post-release: **-0.15%** (continued weakness)
- 5-day announcement return: **-1.86%** (muted reaction)
- 5-day release return: **-4.24%** (disappointing performance)

**Apple iPhone 15:**
- Pre-announcement daily return: **-0.03%** (neutral/weak)
- Announcement to release: **-0.38%** (bearish trend)
- Post-release: **+0.08%** (slight stabilization)
- 5-day announcement return: **-7.06%** (strong negative reaction)
- 5-day release return: **-0.13%** (minor decline)

### 2. Volume Analysis Patterns

**Volume Changes Around Announcements (Baseline-Computed):**

| Product | Pre-Announcement Volume (M) | Announcement to Release (M) | Post-Release Volume (M) |
|---------|----------------------------|----------------------------|------------------------|
| Xbox Series X/S | 35.1 | 31.3 (-11%) | 26.3 (-25%) |
| RTX 30 Series | 423.9 | 831.6 (+96%) | 466.5 (+10%) |
| RTX 40 Series | 546.6 | 613.7 (+12%) | 568.9 (+4%) |
| RTX 40 SUPER | 421.3 | 558.1 (+32%) | 515.3 (+22%) |
| iPhone 12 | 175.3 | 134.7 (-23%) | 108.8 (-38%) |
| iPhone 13 | 78.3 | 91.4 (+17%) | 74.1 (-5%) |
| iPhone 14 | 73.9 | 92.4 (+25%) | 94.0 (+27%) |
| iPhone 15 | 56.3 | 73.2 (+30%) | 55.0 (-2%) |

**Volume Spike Methodology**: Percentages computed as `(period_volume - pre_announcement_baseline) / pre_announcement_baseline * 100`. The formal "volume_spike_pct" metric uses 20-day moving average baseline: `(volume[period] / baseline_MA20) - 1`.

### 3. Trading Signal Identification

**Strong Buy Signals:**
1. **RTX 30 Series**: Strong pre-announcement momentum (+0.70% daily), massive volume spike (+96%), and positive announcement reaction (+8.43%)
2. **RTX 40 SUPER**: Positive momentum building (+2.35% announcement to release), strong announcement reaction (+5.51%), and sustained post-launch performance
3. **iPhone 12**: Strong pre-announcement momentum (+0.48% daily), positive announcement reaction (+7.02%) - **best iPhone signal**

**Bearish Signals:**
1. **RTX 40 Series**: Negative pre-launch momentum (-0.25%), followed by significant post-release sell-off (-12.94%)
2. **iPhone 15**: Negative pre-announcement trend (-0.03% daily), strong negative announcement reaction (-7.06%)
3. **iPhone 13**: Negative announcement reaction (-5.47%) despite moderate pre-momentum

**Mixed/Neutral Signals:**
1. **Xbox Series X/S**: Strong negative announcement reaction (-7.03%) despite positive underlying momentum - **contrarian opportunity**
2. **iPhone 14**: Weak signals across all periods - **avoid or minimal exposure**
3. Products showing classic "buy the rumor, sell the news" patterns

**Key Cross-Company Pattern:**
- **Apple iPhone launches show declining effectiveness over time** (iPhone 12 strong → iPhone 15 weak)
- **NVIDIA shows most consistent high-volatility signals** (both positive and negative extremes)
- **Microsoft shows unique contrarian patterns** (negative announcement, positive recovery)

## Strategic Trading Implications

### 1. Pre-Announcement Positioning Strategy

**Entry Timing:** 30-60 days before announcement
- **Bullish Setup**: Look for stocks showing consistent positive daily returns (>0.5%) with building volume
- **Risk Management**: Exit positions if 5-day announcement return exceeds +/-8% (profit-taking zone)

### 2. Volume-Based Early Warning System

**High Probability Setups:**
- Volume increases >50% in announcement-to-release period typically indicate strong institutional interest
- NVIDIA products showed more pronounced volume patterns than Microsoft (semiconductor vs. consumer hardware difference)

### 3. Post-Announcement Strategy

**"Sell-the-News" Timing:**
- Products with strong pre-announcement momentum often experience profit-taking
- RTX 40 Series showed extreme volatility (-12.94% post-release) suggesting high-risk environment

### 4. Company-Specific Patterns

**Semiconductor Stocks (NVIDIA):**
- Higher volatility and volume spikes
- More pronounced market reactions
- Better suited for momentum strategies
- Consistent high-volatility signals (both positive and negative)

**Tech Conglomerates (Microsoft):**
- More stable price action
- Lower volume impact from single product launches
- Better for longer-term value strategies
- Unique contrarian opportunities (negative announcement → positive recovery)

**Consumer Hardware (Apple iPhone):**
- **Declining market impact over time**: iPhone 12 (+7.02% announcement) → iPhone 15 (-7.06% announcement)
- **Maturation effect**: Later iPhone generations show weaker market reactions
- **Volume patterns**: Generally decreasing volume around announcements (175M → 56M baseline)
- **"iPhone Fatigue"**: Market becoming less responsive to incremental iPhone updates
- **Trading strategy**: Focus on major redesign years, avoid "S" or incremental update cycles

## Methodological Notes and Limitations

### Statistical Considerations
**Multiple Testing**: This analysis examines 8 product launches across multiple metrics without formal multiple-testing corrections. Results should be considered exploratory and require out-of-sample validation.

**Market-Adjusted Returns**: Returns presented are raw stock returns and have not been adjusted for market movements (e.g., relative to XLK/SOXX sector ETFs). Beta effects may influence results during broad market movements.

**Data Adjustments**: All price returns use adjusted close prices from yfinance, accounting for stock splits and dividends. Trading volumes represent raw daily totals without adjustment for stock splits.

### Sample Limitations
**Limited Time Span**: Analysis covers 2020-2024, representing a specific market regime that may not generalize to other periods.

**Company Coverage**: Three companies (Apple, Microsoft, NVIDIA) may not represent broader technology sector patterns.

**Event Selection**: Product launch dates based on public announcements; market may have anticipated some events earlier.

## Risk Management Framework

### 1. Position Sizing Guidelines
- **High-conviction signals**: Max 3-5% of portfolio
- **Speculative plays**: Max 1-2% of portfolio
- **Never risk more than can afford to lose** on single product launch events

### 2. Stop-Loss Levels
- **Tight stops**: 3-5% below entry for momentum plays
- **Wide stops**: 10-15% for longer-term positions
- **Time-based stops**: Exit if thesis doesn't play out within expected timeframe

### 3. Profit-Taking Rules
- **First target**: 5-8% gains (high probability zone)
- **Second target**: 12-15% gains (based on historical maximums)
- **Trail stops**: Implement after reaching first target

## Regulatory and Compliance Considerations

⚠️ **Important Legal Notice:**
- This analysis is for educational purposes only
- Trading based on product launch information may constitute material non-public information in certain jurisdictions
- Always consult with legal counsel before implementing any trading strategies
- Be aware of SEC regulations regarding insider trading and information asymmetry

## Future Research Directions

### 1. Options Flow Analysis
- Examine unusual options activity before announcements
- Analyze put/call ratios and implied volatility patterns
- Study option expiration alignment with product launch dates

### 2. Social Sentiment Integration
- Incorporate social media sentiment analysis
- Track analyst upgrade/downgrade patterns
- Monitor tech blog and industry publication coverage

### 3. Cross-Asset Correlation
- Study competitor stock reactions to announcements
- Analyze sector ETF movements
- Examine supply chain impact (suppliers/partners)

### 4. Machine Learning Enhancement
- Develop predictive models using historical patterns
- Implement real-time anomaly detection for volume/price
- Create automated alert systems for high-probability setups

## Conclusion

The analysis reveals distinct patterns in stock behavior around major technology product launches:

1. **NVIDIA products show higher volatility and clearer trading signals** than Microsoft
2. **Volume spikes are reliable indicators** of institutional positioning
3. **Pre-announcement momentum is crucial** for predicting post-announcement performance
4. **"Sell-the-news" effects are common** but vary significantly by product and market environment

**Risk-Adjusted Recommendation:** Focus on NVIDIA product launches for short-term trading opportunities while using Microsoft launches for longer-term positioning strategies. Always maintain strict risk management protocols and stay informed about regulatory requirements.

---

*This analysis is for educational and research purposes only. Past performance does not guarantee future results. Always conduct your own research and consider your risk tolerance before making investment decisions.*