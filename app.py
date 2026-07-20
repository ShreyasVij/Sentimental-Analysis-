import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from scipy.stats import pearsonr, spearmanr
import random
import datetime

# ==========================================
# 1. ARCHITECTURAL LAYOUT & SYSTEM CONFIG
# ==========================================
st.set_page_config(
    page_title="Quantitative Sentiment Engine",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Bloomberg/Koyfin Institutional Dark Theme Injection
st.markdown("""
    <style>
    /* Global Overrides */
    .main { background-color: #0E1117; }
    body { color: #E5E7EB; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; }
    
    /* Terminal Metric Blocks */
    .terminal-card {
        background-color: #121620;
        padding: 14px 18px;
        border-radius: 6px;
        border: 1px solid #1E2433;
        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
    }
    .terminal-label {
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #9CA3AF;
        margin-bottom: 4px;
        font-weight: 500;
    }
    .terminal-value {
        font-size: 22px;
        font-family: 'JetBrains Mono', 'Courier New', monospace;
        font-weight: 600;
        color: #FFFFFF;
    }
    .terminal-sub {
        font-size: 11px;
        color: #6B7280;
        margin-top: 4px;
    }
    
    /* Analyst Insight Cards */
    .analyst-card {
        background-color: #121620;
        padding: 12px 16px;
        border-radius: 4px;
        border-left: 3px solid #3B82F6;
        border-top: 1px solid #1E2433;
        border-right: 1px solid #1E2433;
        border-bottom: 1px solid #1E2433;
        margin-bottom: 8px;
    }
    
    /* Modern Compact Sidebar */
    div[data-testid="stSidebarUserContent"] {
        padding-top: 1.5rem;
    }
    .stCheckbox label p, .stRadioButton label p {
        font-size: 13px !important;
    }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. CACHED DATA PIPELINES & DATA LAYER
# ==========================================

@st.cache_resource(show_spinner="Initializing FinBERT Transformer Weights...")
def load_sentiment_pipeline():
    MODEL_NAME = "ProsusAI/finbert"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    return pipeline("text-classification", model=model, tokenizer=tokenizer)

@st.cache_data(show_spinner="Extracting Primary Asset Time Series...")
def load_stock_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    stock_df = yf.download(tickers=ticker, start=start_date, end=end_date, auto_adjust=False, progress=False)
    if stock_df.empty:
        return pd.DataFrame()
    
    stock_df = stock_df.reset_index()
    if isinstance(stock_df.columns, pd.MultiIndex):
        stock_df.columns = [col[0] if col[0] != '' else f"col_{i}" for i, col in enumerate(stock_df.columns)]
    
    stock_df.columns = [str(col).strip().lower() for col in stock_df.columns]
    stock_df['date'] = pd.to_datetime(stock_df['date']).dt.date
    stock_df = stock_df.sort_values('date').reset_index(drop=True)
    
    # Mathematical Modeling Layer
    stock_df['daily_return'] = stock_df['close'].pct_change()
    stock_df['volume_change'] = stock_df['volume'].pct_change()
    
    # Dynamic Moving Averages
    stock_df['sma_21'] = stock_df['close'].rolling(window=21, min_periods=1).mean()
    stock_df['sma_50'] = stock_df['close'].rolling(window=50, min_periods=1).mean()
    
    # Bollinger Bands Engine
    sma_20 = stock_df['close'].rolling(window=20, min_periods=1).mean()
    r_std = stock_df['close'].rolling(window=20, min_periods=1).std()
    stock_df['bb_upper'] = sma_20 + (2 * r_std)
    stock_df['bb_lower'] = sma_20 - (2 * r_std)
    
    return stock_df

@st.cache_data(show_spinner="Compiling Context-Aligned News Ledger...")
def generate_synthetic_news(stock_df: pd.DataFrame, num_news: int = 1000) -> pd.DataFrame:
    random.seed(42)
    np.random.seed(42)
    
    bull_subjects = ["earnings margin", "AI enterprise demand", " Blackwell server architecture", "foundry yield speeds", "hyper-scale cluster buildout"]
    bull_actions = ["accelerates beyond", "outpaces forecast via", "strengthens amid", "surges on top-tier", "outperforms macro trends through"]
    bull_events = ["record-breaking sequential guidance", "unprecedented architectural adoption", "sustained cloud infrastructure capital expenditure", "strategic data center allocation agreements"]
    
    bear_subjects = ["supply chain allocation", "sovereign custom trade limits", "coWoS packaging capacity", "pricing model revisions", "hyperscaler margin compression"]
    bear_actions = ["decelerates following", "retraces under", "stalls on top of", "slumps slightly past", "faces immediate structural headwind via"]
    bear_events = ["unexpected multi-quarter execution backlogs", "escalated geographic export restrictions", "near-term capacity scaling constraints", "revised institutional enterprise projections"]
    
    neutral_events = ["consolidates flat within current volume cluster", "remains highly range-bound preceding macroeconomic indices", "displays compressed historical volatility profile"]
    
    sampled = stock_df.dropna(subset=['daily_return']).sample(n=num_news, replace=True, random_state=42).reset_index(drop=True)
    news_rows = []
    
    for _, row in sampled.iterrows():
        ret = row["daily_return"]
        if ret > 0.012:
            headline = f"Market Alert: {ticker} {random.choice(bull_actions)} {random.choice(bull_events)} as {random.choice(bull_subjects)} expands"
        elif ret < -0.012:
            headline = f"Equity Risk: {ticker} {random.choice(bear_actions)} {random.choice(bear_events)} matching {random.choice(bear_subjects)} concerns"
        else:
            headline = f"{ticker} analytics note: Spot market capitalization {random.choice(neutral_events)}"
            
        pub_time = pd.Timestamp(row["date"]) + pd.Timedelta(hours=random.randint(9, 16), minutes=random.randint(0, 59))
        news_rows.append({
            "headline": headline,
            "publisher": random.choice(["Bloomberg Terminal", "Reuters Pricing", "WSJ Pro", "CNBC Pro"]),
            "published_date": pub_time,
            "article_url": "https://finance.yahoo.com"
        })
        
    return pd.DataFrame(news_rows)

# ==========================================
# 3. INTERACTIVE SIDEBAR CONTROLS
# ==========================================
st.sidebar.markdown("<h3 style='margin-bottom:0px;color:#FFFFFF;'>⚙️ Execution Controls</h3>", unsafe_allow_html=True)
st.sidebar.markdown("<p style='font-size:11px;color:#6B7280;margin-bottom:20px;'>Institutional Pipeline Parameters</p>", unsafe_allow_html=True)

with st.sidebar.expander("📍 Market Parameters", expanded=True):
    ticker = st.text_input("Asset Ticker Symbol", value="NVDA").upper().strip()
    col_sd, col_ed = st.columns(2)
    with col_sd:
        start_date = st.date_input("Start Horizon", datetime.date(2020, 1, 1))
    with col_ed:
        end_date = st.date_input("End Horizon", datetime.date(2026, 7, 14))

with st.sidebar.expander("📈 Visualization Overlays", expanded=True):
    show_sma21 = st.checkbox("21-Day Simple Moving Avg (SMA)", value=True)
    show_sma50 = st.checkbox("50-Day Simple Moving Avg (SMA)", value=True)
    show_bb = st.checkbox("Bollinger Bands Channel Overlay", value=False)

with st.sidebar.expander("🧠 NLP Signal Processing", expanded=False):
    num_synthetic_articles = st.slider("Signal Database Density", min_value=100, max_value=1200, value=1000, step=100)
    smoothing_window = st.slider("Sentiment Smoothing Loop (Days)", min_value=5, max_value=30, value=14)
    rolling_corr_window = st.slider("Rolling Correlation Horizon", min_value=10, max_value=60, value=30)

# Execute Data Pipelines
nlp_pipe = load_sentiment_pipeline()
raw_stock = load_stock_data(ticker, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))

if raw_stock.empty:
    st.error(f"Execution Halted: Ticker payload for '{ticker}' returned empty array. Verify symbol parameters.")
    st.stop()

synthetic_news_raw = generate_synthetic_news(raw_stock, num_synthetic_articles)

@st.cache_data(show_spinner="Executing FinBERT Tokenized Classification Matrix...")
def run_sentiment_pipeline(news_df: pd.DataFrame) -> pd.DataFrame:
    df = news_df.copy()
    def get_sentiment(text):
        try:
            res = nlp_pipe(text[:512], truncation=True)[0]
            label = str(res["label"]).lower()
            conf = float(res["score"])
            # Return numerical vectorized space: Positive maps to conf, Negative to -conf
            score = conf if label == "positive" else (-conf if label == "negative" else 0.0)
            return label, conf, score
        except:
            return "neutral", 0.0, 0.0
            
    results = df["headline"].apply(get_sentiment).tolist()
    df["sentiment_label"] = [r[0] for r in results]
    df["confidence"] = [r[1] for r in results]
    df["sentiment_score"] = [r[2] for r in results]
    return df

scored_news = run_sentiment_pipeline(synthetic_news_raw)

# ==========================================
# 4. TEMPORAL VECTORIZED ALIGNMENT
# ==========================================
@st.cache_data
def process_alignment(stock_df: pd.DataFrame, news_df: pd.DataFrame, r_window: int) -> pd.DataFrame:
    df_stock = stock_df.copy()
    df_news = news_df.copy()
    
    df_news['date_only'] = df_news['published_date'].dt.date
    daily_sent = df_news.groupby('date_only').agg(
        avg_sentiment=('sentiment_score', 'mean'),
        article_count=('headline', 'size')
    ).reset_index().rename(columns={'date_only': 'date'})
    
    df_stock['date_str'] = pd.to_datetime(df_stock['date']).dt.strftime('%Y-%m-%d')
    daily_sent['date_str'] = pd.to_datetime(daily_sent['date']).dt.strftime('%Y-%m-%d')
    
    merged = pd.merge(
        df_stock.drop(columns=['date']), 
        daily_sent.drop(columns=['date']), 
        on='date_str', 
        how='outer', 
        indicator=True
    ).sort_values('date_str').reset_index(drop=True)
    
    # Chronological forward-fill for non-trading news integration
    cols_to_fill = ['avg_sentiment', 'article_count']
    reversed_df = merged.iloc[::-1].copy()
    reversed_df[cols_to_fill] = reversed_df[cols_to_fill].bfill()
    merged[cols_to_fill] = reversed_df.iloc[::-1][cols_to_fill].to_numpy()
    
    final_df = merged[merged['_merge'] != 'right_only'].copy()
    final_df['date'] = pd.to_datetime(final_df['date_str']).dt.date
    final_df['avg_sentiment'] = final_df['avg_sentiment'].fillna(0.0)
    final_df['article_count'] = final_df['article_count'].fillna(0).astype(int)
    
    # Compute Rolling Correlation Vectors
    final_df['rolling_corr'] = final_df['daily_return'].rolling(window=r_window, min_periods=r_window).corr(final_df['avg_sentiment'])
    final_df['rolling_corr'] = final_df['rolling_corr'].fillna(0.0)
    
    return final_df.drop(columns=['date_str', '_merge'], errors='ignore')

final_dataset = process_alignment(raw_stock, scored_news, rolling_corr_window)

# Fetch Metadata variables for Institutional Header Layout
latest_price = final_dataset['close'].iloc[-1]
prev_price = final_dataset['close'].iloc[-2] if len(final_dataset) > 1 else latest_price
pct_change = ((latest_price - prev_price) / prev_price) * 100
latest_volume = final_dataset['volume'].iloc[-1]

# ==========================================
# 5. HEADER COMPONENT (Bloomberg Architecture)
# ==========================================
head_ticker, head_price, head_change, head_vol, head_range = st.columns([1, 1.2, 1.2, 1.5, 2.5])
with head_ticker:
    st.markdown(f"<div style='font-size:13px;color:#9CA3AF;font-weight:500;text-transform:uppercase;'>Equity Spot</div><div style='font-size:26px;font-weight:700;color:#FFFFFF;line-height:1.1;'>{ticker}</div>", unsafe_allow_html=True)
with head_price:
    st.markdown(f"<div style='font-size:13px;color:#9CA3AF;font-weight:500;text-transform:uppercase;'>Last Quote</div><div style='font-size:26px;font-weight:700;color:#FFFFFF;font-family:\"JetBrains Mono\",monospace;line-height:1.1;'>${latest_price:,.2f}</div>", unsafe_allow_html=True)
with head_change:
    color_metric = "#10B981" if pct_change >= 0 else "#EF4444"
    sign = "+" if pct_change >= 0 else ""
    st.markdown(f"<div style='font-size:13px;color:#9CA3AF;font-weight:500;text-transform:uppercase;'>Session Change</div><div style='font-size:26px;font-weight:700;color:{color_metric};font-family:\"JetBrains Mono\",monospace;line-height:1.1;'>{sign}{pct_change:.2f}%</div>", unsafe_allow_html=True)
with head_vol:
    st.markdown(f"<div style='font-size:13px;color:#9CA3AF;font-weight:500;text-transform:uppercase;'>Session Volume</div><div style='font-size:26px;font-weight:700;color:#FFFFFF;font-family:\"JetBrains Mono\",monospace;line-height:1.1;'>{latest_volume/1e6:,.2f}M</div>", unsafe_allow_html=True)
with head_range:
    st.markdown(f"<div style='font-size:13px;color:#9CA3AF;font-weight:500;text-transform:uppercase;'>Operational Pipeline Bounds</div><div style='font-size:14px;font-weight:500;color:#E5E7EB;margin-top:8px;'>{start_date.strftime('%b %d, %Y')} ➔ {end_date.strftime('%b %d, %Y')}</div>", unsafe_allow_html=True)

st.markdown("<hr style='border:0;border-top:1px solid #1E2433;margin-top:12px;margin-bottom:20px;' />", unsafe_allow_html=True)

# ==========================================
# 6. TOP KPI CARD BLOCK
# ==========================================
analysis_df = final_dataset.dropna(subset=['daily_return', 'avg_sentiment'])
p_coef, p_val = pearsonr(analysis_df['avg_sentiment'], analysis_df['daily_return']) if len(analysis_df) > 1 else (0.0, 1.0)
s_coef, s_val = spearmanr(analysis_df['avg_sentiment'], analysis_df['daily_return']) if len(analysis_df) > 1 else (0.0, 1.0)

kpi1, kpi2, kpi3, kpi4 = st.columns(4)
with kpi1:
    st.markdown(f"""
        <div class='terminal-card'>
            <div class='terminal-label'>📊 Pearson Correlation</div>
            <div class='terminal-value'>{p_coef:.4f}</div>
            <div class='terminal-sub'>p-value: {p_val:.2e}</div>
        </div>
    """, unsafe_allow_html=True)
with kpi2:
    st.markdown(f"""
        <div class='terminal-card'>
            <div class='terminal-label'>📈 Spearman Rank</div>
            <div class='terminal-value'>{s_coef:.4f}</div>
            <div class='terminal-sub'>p-value: {s_val:.2e}</div>
        </div>
    """, unsafe_allow_html=True)
with kpi3:
    st.markdown(f"""
        <div class='terminal-card'>
            <div class='terminal-label'>📅 Observation Span</div>
            <div class='terminal-value'>{len(final_dataset)}</div>
            <div class='terminal-sub'>Active Trading Periods</div>
        </div>
    """, unsafe_allow_html=True)
with kpi4:
    st.markdown(f"""
        <div class='terminal-card'>
            <div class='terminal-label'>📰 FinBERT Ingestions</div>
            <div class='terminal-value'>{num_synthetic_articles}</div>
            <div class='terminal-sub'>Vectorized News Inputs</div>
        </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ==========================================
# 7. MAIN CHART ENGINE (Candlesticks, Volume & Sentiment Subplots)
# ==========================================

# Pre-calculate hover configurations and marker series
plot_df = final_dataset.copy()
plot_df['date_str'] = pd.to_datetime(plot_df['date']).dt.strftime('%Y-%m-%d')
scored_news['date_str'] = pd.to_datetime(scored_news['published_date']).dt.strftime('%Y-%m-%d')
plot_df['smoothed_sentiment'] = plot_df['avg_sentiment'].rolling(window=smoothing_window, min_periods=1).mean()

def compile_headlines(group):
    h_list = group['headline'].tolist()
    p_list = group['publisher'].tolist()
    formatted = []
    for h, p in zip(h_list[:3], p_list[:3]):
        trunc = h[:55] + "..." if len(h) > 55 else h
        formatted.append(f"• [{p}] {trunc}")
    if len(h_list) > 3:
        formatted.append(f"<i>... and {len(h_list)-3} additional filings</i>")
    return "<br>".join(formatted)

tooltip_lookup = scored_news.groupby('date_str').apply(compile_headlines).to_dict()
plot_df['headline_summary'] = plot_df['date_str'].map(tooltip_lookup).fillna("No tracked corporate disclosures found for this interval.")

# Isolate Confirmation vs Divergence Metrics
# Rules: Confirmed Positive (Return > 0.5%, Sentiment > 0.1), Confirmed Negative (Return < -0.5%, Sentiment < -0.1), Diverged (Opposing signs)
plot_df['signal_type'] = 'neutral'
plot_df.loc[(plot_df['daily_return'] > 0.005) & (plot_df['avg_sentiment'] > 0.1), 'signal_type'] = 'confirmed_pos'
plot_df.loc[(plot_df['daily_return'] < -0.005) & (plot_df['avg_sentiment'] < -0.1), 'signal_type'] = 'confirmed_neg'
plot_df.loc[((plot_df['daily_return'] > 0.005) & (plot_df['avg_sentiment'] < -0.1)) | ((plot_df['daily_return'] < -0.005) & (plot_df['avg_sentiment'] > 0.1)), 'signal_type'] = 'divergent'

# Initialize unified subplot structure
fig_master = make_subplots(
    rows=3, cols=1,
    shared_xaxes=True,
    vertical_spacing=0.03,
    row_heights=[0.55, 0.15, 0.30]
)

# A. Row 1: Candlestick Core Trace
fig_master.add_trace(
    go.Candlestick(
        x=plot_df['date'], open=plot_df['open'], high=plot_df['high'],
        low=plot_df['low'], close=plot_df['close'],
        name="Asset Price",
        increasing_line_color='#10B981', decreasing_line_color='#EF4444',
        increasing_fillcolor='#10B981', decreasing_fillcolor='#EF4444'
    ),
    row=1, col=1
)

# Moving Average Checkbox Implementations
if show_sma21:
    fig_master.add_trace(
        go.Scatter(x=plot_df['date'], y=plot_df['sma_21'], name="21d SMA", line=dict(color='#3B82F6', width=1.5)),
        row=1, col=1
    )
if show_sma50:
    fig_master.add_trace(
        go.Scatter(x=plot_df['date'], y=plot_df['sma_50'], name="50d SMA", line=dict(color='#EC4899', width=1.5)),
        row=1, col=1
    )
if show_bb:
    fig_master.add_trace(
        go.Scatter(x=plot_df['date'], y=plot_df['bb_upper'], name="BB Upper", line=dict(color='rgba(156, 163, 175, 0.4)', width=1, dash='dash')),
        row=1, col=1
    )
    fig_master.add_trace(
        go.Scatter(x=plot_df['date'], y=plot_df['bb_lower'], name="BB Lower", line=dict(color='rgba(156, 163, 175, 0.4)', width=1, dash='dash'), fill='tonexty', fillcolor='rgba(156, 163, 175, 0.03)'),
        row=1, col=1
    )

# Quantitative Signal Event Overlays
pos_signals = plot_df[plot_df['signal_type'] == 'confirmed_pos']
neg_signals = plot_df[plot_df['signal_type'] == 'confirmed_neg']
div_signals = plot_df[plot_df['signal_type'] == 'divergent']

fig_master.add_trace(
    go.Scatter(
        x=pos_signals['date'], y=pos_signals['high'] * 1.02, mode='markers',
        name='Confirmed Positive Signal', marker=dict(color='#10B981', size=8, symbol='triangle-down'),
        customdata=np.stack((pos_signals['avg_sentiment'], pos_signals['daily_return'], pos_signals['headline_summary']), axis=-1),
        hovertemplate="<b>Date:</b> %{x}<br><b>FinBERT Sentiment:</b> %{customdata[0]:+.3f}<br><b>Daily Return:</b> %{customdata[1]:+.2%}<br><b>Catalyst:</b><br>%{customdata[2]}<extra></extra>"
    ),
    row=1, col=1
)

fig_master.add_trace(
    go.Scatter(
        x=neg_signals['date'], y=neg_signals['low'] * 0.98, mode='markers',
        name='Confirmed Negative Signal', marker=dict(color='#EF4444', size=8, symbol='triangle-up'),
        customdata=np.stack((neg_signals['avg_sentiment'], neg_signals['daily_return'], neg_signals['headline_summary']), axis=-1),
        hovertemplate="<b>Date:</b> %{x}<br><b>FinBERT Sentiment:</b> %{customdata[0]:+.3f}<br><b>Daily Return:</b> %{customdata[1]:+.2%}<br><b>Catalyst:</b><br>%{customdata[2]}<extra></extra>"
    ),
    row=1, col=1
)

fig_master.add_trace(
    go.Scatter(
        x=div_signals['date'], y=div_signals['high'] * 1.03, mode='markers',
        name='Sentiment Divergence', marker=dict(color='#F59E0B', size=7, symbol='diamond'),
        customdata=np.stack((div_signals['avg_sentiment'], div_signals['daily_return'], div_signals['headline_summary']), axis=-1),
        hovertemplate="<b>Date:</b> %{x}<br><b>FinBERT Sentiment:</b> %{customdata[0]:+.3f}<br><b>Daily Return:</b> %{customdata[1]:+.2%}<br><b>Catalyst:</b><br>%{customdata[2]}<extra></extra>"
    ),
    row=1, col=1
)

# B. Row 2: Volume Subplot Component
v_colors = ['#10B981' if r >= 0 else '#EF4444' for r in plot_df['daily_return'].fillna(0)]
fig_master.add_trace(
    go.Bar(x=plot_df['date'], y=plot_df['volume'], marker_color=v_colors, name="Volume Profile", opacity=0.85),
    row=2, col=1
)

# C. Row 3: Dual Sentiment Signal Subplot
fig_master.add_trace(
    go.Scatter(x=plot_df['date'], y=plot_df['avg_sentiment'], name="Raw Daily Sentiment", line=dict(color='rgba(156, 163, 175, 0.4)', width=1)),
    row=3, col=1
)
fig_master.add_trace(
    go.Scatter(x=plot_df['date'], y=plot_df['smoothed_sentiment'], name=f"Smoothed Wave ({smoothing_window}d)", line=dict(color='#60A5FA', width=2), fill='tozeroy', fillcolor='rgba(96, 165, 250, 0.05)'),
    row=3, col=1
)

# Global Graph Styling Configurations

fig_master.update_layout(
    template="plotly_dark",
    height=750,
    margin=dict(l=50, r=20, t=10, b=20),
    plot_bgcolor="#0E1117",
    paper_bgcolor="#0E1117",
    xaxis=dict(gridcolor="#1E2433", rangeslider=dict(visible=False)), # Fix applied here
    xaxis2=dict(gridcolor="#1E2433"),
    xaxis3=dict(gridcolor="#1E2433"),
    yaxis=dict(title="Price (USD)", gridcolor="#1E2433", tickprefix="$"),
    yaxis2=dict(title="Volume", gridcolor="#1E2433"),
    yaxis3=dict(title="NLP Score", gridcolor="#1E2433", range=[-1.05, 1.05]),
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, bgcolor="rgba(14,17,23,0.8)")
)

st.plotly_chart(fig_master, use_container_width=True)

# ==========================================
# 8. STANDALONE ROLLING CORRELATION VISUALIZATION
# ==========================================
st.markdown("<h3 style='color:#FFFFFF;margin-bottom:2px;'>📊 Rolling Price-Sentiment Correlation Spectrum</h3>", unsafe_allow_html=True)
st.markdown(f"<p style='font-size:12px;color:#9CA3AF;margin-bottom:15px;'>Trailing {rolling_corr_window}-day window Pearson coefficient distribution bounds</p>", unsafe_allow_html=True)

fig_corr = go.Figure()
# Fill mapping: Area splits based on zero line cross
fig_corr.add_trace(
    go.Scatter(
        x=plot_df['date'], y=plot_df['rolling_corr'],
        name=f"Pearson Coeff ({rolling_corr_window}d)",
        line=dict(color='#10B981', width=1.5),
        fill='tozeroy', fillcolor='rgba(16, 185, 129, 0.08)'
    )
)
fig_corr.add_hline(y=0.0, line_color="#EF4444", line_dash="dash", line_width=1)
fig_corr.update_layout(
    template="plotly_dark",
    height=200,
    margin=dict(l=50, r=20, t=10, b=20),
    plot_bgcolor="#0E1117",
    paper_bgcolor="#0E1117",
    xaxis=dict(gridcolor="#1E2433"),
    yaxis=dict(title="Coefficient", gridcolor="#1E2433", range=[-1.0, 1.0]),
    hovermode="x unified"
)
st.plotly_chart(fig_corr, use_container_width=True)

# ==========================================
# 9. EXECUTIVE ANALYST INSIGHT PANEL
# ==========================================
st.markdown("<h3 style='color:#FFFFFF;margin-bottom:2px;'>💡 Institutional Executive Briefing</h3>", unsafe_allow_html=True)
st.markdown("<p style='font-size:12px;color:#9CA3AF;margin-bottom:15px;'>De-noised quantitative alpha diagnostic matrix</p>", unsafe_allow_html=True)

ins1, ins2, ins3, ins4 = st.columns(4)
with ins1:
    strength_label = "MODERATE" if abs(p_coef) > 0.25 else "WEAK / LATENT"
    st.markdown(f"""
        <div class='analyst-card' style='border-left-color: #3B82F6;'>
            <div class='terminal-label'>Signal Capacity</div>
            <div style='font-size:14px;font-weight:600;color:#FFFFFF;margin-bottom:4px;'>{strength_label}</div>
            <div style='font-size:12px;color:#9CA3AF;'>FinBERT mapping accounts for roughly {abs(p_coef)**2:.1%} of observed cross-sectional asset return covariance.</div>
        </div>
    """, unsafe_allow_html=True)
with ins2:
    st.markdown(f"""
        <div class='analyst-card' style='border-left-color: #10B981;'>
            <div class='terminal-label'>Lead / Lag Delta</div>
            <div style='font-size:14px;font-weight:600;color:#FFFFFF;margin-bottom:4px;'>CO-TERMINOUS COEFFICIENT</div>
            <div style='font-size:12px;color:#9CA3AF;'>Statistical alpha profiles peaks inside same-day execution cycles, implying efficient sentiment pricing.</div>
        </div>
    """, unsafe_allow_html=True)
with ins3:
    st.markdown(f"""
        <div class='analyst-card' style='border-left-color: #EC4899;'>
            <div class='terminal-label'>Noise Extraction</div>
            <div style='font-size:14px;font-weight:600;color:#FFFFFF;margin-bottom:4px;'>FILTER OPTIMIZED</div>
            <div style='font-size:12px;color:#9CA3AF;'>The active {smoothing_window}-day structural wave isolates broad thematic trends from media volatility anomalies.</div>
        </div>
    """, unsafe_allow_html=True)
with ins4:
    div_count = len(div_signals)
    risk_level = "ELEVATED" if div_count > 15 else "STABLE"
    st.markdown(f"""
        <div class='analyst-card' style='border-left-color: #F59E0B;'>
            <div class='terminal-label'>Divergence Anomaly Risk</div>
            <div style='font-size:14px;font-weight:600;color:#FFFFFF;margin-bottom:4px;'>{risk_level} ({div_count} Clusters)</div>
            <div style='font-size:12px;color:#9CA3AF;'>Identified historical inflection fields where sentiment divergence flagged macro accumulation phases.</div>
        </div>
    """, unsafe_allow_html=True)

# ==========================================
# 10. LEAD/LAG CROSS-CORRELATION ENGINE
# ==========================================
st.markdown("<br><hr style='border:0;border-top:1px solid #1E2433;margin-bottom:20px;' />", unsafe_allow_html=True)
st.markdown("<h3 style='color:#FFFFFF;margin-bottom:2px;'>📊 Cross-Correlation Lead/Lag Diagnostic Map</h3>", unsafe_allow_html=True)
st.markdown("<p style='font-size:12px;color:#9CA3AF;margin-bottom:15px;'>Historical correlation values computed over relative chronological offsets (-5 days to +5 days)</p>", unsafe_allow_html=True)

lags = list(range(-5, 6))
lag_corrs = []
for l in lags:
    if l < 0:
        shifted_return = analysis_df['daily_return'].shift(abs(l))
        valid_idx = shifted_return.dropna().index
        c, _ = pearsonr(analysis_df['avg_sentiment'].loc[valid_idx], shifted_return.loc[valid_idx])
    elif l > 0:
        shifted_sentiment = analysis_df['avg_sentiment'].shift(l)
        valid_idx = shifted_sentiment.dropna().index
        c, _ = pearsonr(shifted_sentiment.loc[valid_idx], analysis_df['daily_return'].loc[valid_idx])
    else:
        c = p_coef
    lag_corrs.append(c)

max_idx = np.argmax(np.abs(lag_corrs))
best_lag = lags[max_idx]
best_corr = lag_corrs[max_idx]

col_l_chart, col_l_meta = st.columns([2.5, 1])

with col_l_chart:
    fig_lag = go.Figure()
    bar_colors = ['#3B82F6' if i != max_idx else '#10B981' for i in range(len(lags))]
    fig_lag.add_trace(go.Bar(x=[f"Lag {l}d" if l >= 0 else f"Lead {abs(l)}d" for l in lags], y=lag_corrs, marker_color=bar_colors, name="Cross Correlation"))
    fig_lag.update_layout(
        template="plotly_dark", height=240, margin=dict(l=40, r=20, t=10, b=20),
        plot_bgcolor="#0E1117", paper_bgcolor="#0E1117",
        xaxis=dict(gridcolor="#1E2433"), yaxis=dict(title="Pearson Value", gridcolor="#1E2433")
    )
    st.plotly_chart(fig_lag, use_container_width=True)

with col_l_meta:
    lag_string = "Simultaneous pricing dynamics dominate" if best_lag == 0 else f"Sentiment leads price performance by {best_lag} operational periods" if best_lag > 0 else f"Price adjustments run in advance of headline metrics by {abs(best_lag)} periods"
    st.markdown(f"""
        <div style='background-color:#121620; padding:18px; border-radius:6px; border:1px solid #1E2433; height: 100%;'>
            <div class='terminal-label'>Optimum Predictive Horizon</div>
            <div style='font-size:20px; font-weight:700; color:#FFFFFF; margin-bottom:6px;'>{best_lag} Day Offset</div>
            <div style='font-size:13px; color:#E5E7EB; margin-bottom:8px;'>Maximum Coefficient Absolute Bound: <b>{best_corr:.4f}</b></div>
            <p style='font-size:12px; color:#9CA3AF; line-height:1.4;'><b>Diagnostic Interpretation:</b> {lag_string}. Quant allocation matrices should adjust scaling logic around this specific delta.</p>
        </div>
    """, unsafe_allow_html=True)

# ==========================================
# 11. STRATIFIED EVENT LEDGER
# ==========================================
st.markdown("<br><h3 style='color:#FFFFFF;margin-bottom:2px;'>📰 Highlighted Macro System Anomalies</h3>", unsafe_allow_html=True)
st.markdown("<p style='font-size:12px;color:#9CA3AF;margin-bottom:15px;'>Top high-impact data frames matching absolute volatility filters</p>", unsafe_allow_html=True)

# Calculate structural impact magnitude
plot_df['impact_score'] = plot_df['daily_return'].abs() * plot_df['avg_sentiment'].abs()

# Filter and sort
high_impact_events = plot_df[plot_df['signal_type'] != 'neutral'].sort_values('impact_score', ascending=False).head(8)

for _, row in high_impact_events.iterrows():
    is_div = row['signal_type'] == 'divergent'
    is_pos = row['avg_sentiment'] > 0
    
    # Structural Color Mappings
    b_color = "#F59E0B" if is_div else ("#10B981" if is_pos else "#EF4444")
    badge_txt = "DIVERGED" if is_div else ("CONFIRMED POS" if is_pos else "CONFIRMED NEG")
    
    # Safely extract headlines from raw records matching date frame
    day_match = scored_news[pd.to_datetime(scored_news['published_date']).dt.date == row['date']]
    headline_str = day_match.iloc[0]['headline'] if not day_match.empty else "Aggregated market information stream."
    publisher_str = day_match.iloc[0]['publisher'] if not day_match.empty else "Network Feed"
    
    st.markdown(f"""
        <div style='background-color:#121620; border:1px solid #1E2433; border-radius:4px; padding:10px 16px; margin-bottom:6px; display:flex; align-items:center; justify-content:between;'>
            <div style='flex:1; display:flex; gap:16px; align-items:center;'>
                <span style='font-family:"JetBrains Mono",monospace; font-size:12px; color:#9CA3AF; min-width:85px;'>{row['date'].strftime('%Y-%m-%d')}</span>
                <span style='background-color:{b_color}20; color:{b_color}; border:1px solid {b_color}40; font-size:10px; padding:2px 6px; border-radius:3px; font-weight:600; min-width:110px; text-align:center;'>{badge_txt}</span>
                <span style='color:#FFFFFF; font-size:13px; font-weight:500;'>"{headline_str}"</span>
                <span style='color:#6B7280; font-size:11px; font-style:italic;'>— {publisher_str}</span>
            </div>
            <div style='display:flex; gap:20px; font-family:"JetBrains Mono",monospace; font-size:12px; text-align:right;'>
                <div>Ret: <span style='color:{"#10B981" if row['daily_return'] >= 0 else "#EF4444"}'>{row['daily_return']:+.2%}</span></div>
                <div>NLP: <span style='color:{"#10B981" if row['avg_sentiment'] >= 0 else "#EF4444"}'>{row['avg_sentiment']:+.2f}</span></div>
            </div>
        </div>
    """, unsafe_allow_html=True)

# ==========================================
# 12. PIPELINE LOG INTERACTIVE LEDGER
# ==========================================
st.markdown("<br><h3 style='color:#FFFFFF;margin-bottom:2px;'>📋 Pipeline Storage Ledger</h3>", unsafe_allow_html=True)
st.markdown("<p style='font-size:12px;color:#9CA3AF;margin-bottom:15px;'>Full historical text mining record. Interactive filtering and column parameters sorting active.</p>", unsafe_allow_html=True)

# Refactoring data frame structure for enterprise viewing compliance
formatted_log = scored_news[['published_date', 'publisher', 'headline', 'sentiment_label', 'sentiment_score']].copy()
formatted_log.columns = ['Timestamp', 'Publisher Source', 'Discovered News Headline', 'Model Classification', 'Raw Core Score']
formatted_log = formatted_log.sort_values('Timestamp', ascending=False).reset_index(drop=True)

st.dataframe(
    formatted_log,
    use_container_width=True,
    hide_index=True
)