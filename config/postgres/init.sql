-- config/postgres/init.sql

CREATE TABLE IF NOT EXISTS raw_price_data (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL,
    symbol VARCHAR(10) NOT NULL DEFAULT 'BTC',
    price_usd DECIMAL(15,8) NOT NULL,
    volume_24h DECIMAL(20,8),
    market_cap DECIMAL(20,8),
    price_change_1h DECIMAL(10,6),
    price_change_24h DECIMAL(10,6),
    price_change_7d DECIMAL(10,6),
    source VARCHAR(50) NOT NULL,
    metadata JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    
    UNIQUE(timestamp, symbol, source)
);

CREATE INDEX IF NOT EXISTS idx_price_data_timestamp ON raw_price_data(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_price_data_symbol_time ON raw_price_data(symbol, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_price_data_source ON raw_price_data(source);

CREATE TABLE IF NOT EXISTS market_sentiment (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL,
    source VARCHAR(50) NOT NULL,
    platform VARCHAR(50) NOT NULL,
    
    sentiment_score DECIMAL(3,2),
    confidence_score DECIMAL(3,2),
    
    mention_count INTEGER DEFAULT 0,
    engagement_score DECIMAL(10,2),
    
    title TEXT,
    content_snippet TEXT,
    url TEXT,
    author VARCHAR(100),
    
    metadata JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_sentiment_timestamp ON market_sentiment(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_sentiment_source ON market_sentiment(source, timestamp DESC);

CREATE TABLE IF NOT EXISTS technical_indicators (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL,
    symbol VARCHAR(10) NOT NULL DEFAULT 'BTC',
    
    sma_20 DECIMAL(15,8),
    sma_50 DECIMAL(15,8),
    sma_200 DECIMAL(15,8),
    ema_12 DECIMAL(15,8),
    ema_26 DECIMAL(15,8),
    
    rsi_14 DECIMAL(5,2),
    macd DECIMAL(15,8),
    macd_signal DECIMAL(15,8),
    macd_histogram DECIMAL(15,8),
    
    bb_upper DECIMAL(15,8),
    bb_middle DECIMAL(15,8),
    bb_lower DECIMAL(15,8),
    atr_14 DECIMAL(15,8),
    
    volume_sma_20 DECIMAL(20,8),
    obv DECIMAL(25,8),
    
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(timestamp, symbol)
);

CREATE INDEX IF NOT EXISTS idx_technical_timestamp ON technical_indicators(timestamp DESC);

CREATE TABLE IF NOT EXISTS features (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL,
    symbol VARCHAR(10) NOT NULL DEFAULT 'BTC',
    
    price_usd DECIMAL(15,8) NOT NULL,
    volume_24h DECIMAL(20,8),
    market_cap DECIMAL(20,8),
    
    returns_1h DECIMAL(10,6),
    returns_24h DECIMAL(10,6),
    returns_7d DECIMAL(10,6),
    log_returns DECIMAL(10,6),
    price_change_pct_1h DECIMAL(8,4),
    price_change_pct_24h DECIMAL(8,4),
    
    volatility_1h DECIMAL(10,6),
    volatility_24h DECIMAL(10,6),
    volatility_7d DECIMAL(10,6),
    realized_volatility DECIMAL(10,6),
    
    sma_5 DECIMAL(15,8),
    sma_10 DECIMAL(15,8),
    sma_20 DECIMAL(15,8),
    sma_50 DECIMAL(15,8),
    sma_200 DECIMAL(15,8),
    ema_12 DECIMAL(15,8),
    ema_26 DECIMAL(15,8),
    ema_50 DECIMAL(15,8),
    
    rsi_14 DECIMAL(5,2),
    rsi_6 DECIMAL(5,2),
    macd DECIMAL(15,8),
    macd_signal DECIMAL(15,8),
    macd_histogram DECIMAL(15,8),
    stoch_k DECIMAL(5,2),
    stoch_d DECIMAL(5,2),
    williams_r DECIMAL(5,2),
    
    bb_upper DECIMAL(15,8),
    bb_middle DECIMAL(15,8),
    bb_lower DECIMAL(15,8),
    bb_width DECIMAL(10,6),
    bb_position DECIMAL(5,2),
    atr_14 DECIMAL(15,8),
    
    volume_sma_20 DECIMAL(20,8),
    volume_ratio DECIMAL(8,4),
    obv DECIMAL(25,8),
    vwap DECIMAL(15,8),
    
    price_position_24h DECIMAL(5,2),
    price_position_7d DECIMAL(5,2),
    price_to_sma20 DECIMAL(6,4),
    price_to_sma50 DECIMAL(6,4),
    sma_ratio_20_50 DECIMAL(6,4),
    
    price_lag_1h DECIMAL(15,8),
    price_lag_3h DECIMAL(15,8),
    price_lag_6h DECIMAL(15,8),
    price_lag_12h DECIMAL(15,8),
    price_lag_24h DECIMAL(15,8),
    rsi_lag_6h DECIMAL(5,2),
    volume_lag_24h DECIMAL(20,8),
    
    price_mean_6h DECIMAL(15,8),
    price_mean_12h DECIMAL(15,8),
    price_mean_24h DECIMAL(15,8),
    price_std_24h DECIMAL(15,8),
    price_min_24h DECIMAL(15,8),
    price_max_24h DECIMAL(15,8),
    
    news_sentiment_1h DECIMAL(3,2),
    news_sentiment_24h DECIMAL(3,2),
    social_sentiment_1h DECIMAL(3,2),
    social_sentiment_24h DECIMAL(3,2),
    sentiment_momentum DECIMAL(3,2),
    fear_greed_index INTEGER,
    
    support_level_1 DECIMAL(15,8),
    support_level_2 DECIMAL(15,8),
    resistance_level_1 DECIMAL(15,8),
    resistance_level_2 DECIMAL(15,8),
    trend_direction INTEGER,
    trend_strength DECIMAL(3,2),
    
    dxy_index DECIMAL(8,4),
    gold_price DECIMAL(10,2),
    sp500_close DECIMAL(10,2),
    vix_index DECIMAL(6,2),
    
    target_price_1h DECIMAL(15,8),
    target_price_24h DECIMAL(15,8),
    target_return_1h DECIMAL(8,4),
    target_return_24h DECIMAL(8,4),
    target_direction_1h INTEGER,
    target_direction_24h INTEGER,
    target_volatility_24h DECIMAL(10,6),
    
    data_quality_score DECIMAL(3,2),
    feature_version VARCHAR(10) DEFAULT '1.0',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    
    UNIQUE(timestamp, symbol)
);

CREATE INDEX IF NOT EXISTS idx_features_timestamp ON features(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_features_symbol_time ON features(symbol, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_features_target_1h ON features(target_return_1h) WHERE target_return_1h IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_features_target_24h ON features(target_return_24h) WHERE target_return_24h IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_features_recent ON features(timestamp DESC) WHERE timestamp > NOW() - INTERVAL '30 days';
CREATE INDEX IF NOT EXISTS idx_features_training ON features(timestamp DESC) WHERE target_return_1h IS NOT NULL;