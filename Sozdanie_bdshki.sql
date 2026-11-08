-- Таблица 1: RusSentiment

CREATE TABLE IF NOT EXISTS rusentiment_posts (
    id          BIGSERIAL PRIMARY KEY,
    label       VARCHAR(20) NOT NULL,
    text        TEXT        NOT NULL,
    created_at  TIMESTAMPTZ DEFAULT NOW(),

    CONSTRAINT chk_label CHECK (
        label IN ('neutral','negative','positive','skip','speech')
    )
);

-- Индексы для частых запросов
CREATE INDEX IF NOT EXISTS idx_rusentiment_label ON rusentiment_posts(label);
CREATE INDEX IF NOT EXISTS idx_rusentiment_text_fts
    ON rusentiment_posts
    USING GIN (to_tsvector('russian', text));

-- Таблица 2: Helpdesk (тикеты службы поддержки, Индонезия)

CREATE TABLE IF NOT EXISTS helpdesk_tickets (
    id              BIGSERIAL PRIMARY KEY,
    ticket_number   VARCHAR(30) NOT NULL UNIQUE,
    request_date    TIMESTAMP   NOT NULL,
    rate_user_notes TEXT,
    final_text_final TEXT,
    sentimen_final  VARCHAR(10) NOT NULL,
    rating_5        SMALLINT    NOT NULL,
    rating_label    VARCHAR(15) NOT NULL,
    created_at      TIMESTAMPTZ DEFAULT NOW(),

    CONSTRAINT chk_sentimen_final
        CHECK (sentimen_final IN ('positif','negatif','netral')),
    CONSTRAINT chk_rating5
        CHECK (rating_5 BETWEEN 1 AND 5),
    CONSTRAINT chk_rating_label
        CHECK (rating_label IN ('puas','tidak puas','netral'))
);

CREATE INDEX IF NOT EXISTS idx_helpdesk_sentimen   ON helpdesk_tickets(sentimen_final);
CREATE INDEX IF NOT EXISTS idx_helpdesk_rating     ON helpdesk_tickets(rating_5);
CREATE INDEX IF NOT EXISTS idx_helpdesk_date       ON helpdesk_tickets(request_date);
CREATE INDEX IF NOT EXISTS idx_helpdesk_text_fts
    ON helpdesk_tickets
    USING GIN (to_tsvector('simple', COALESCE(final_text_final, '')));


-- Полезные представления (views)

CREATE OR REPLACE VIEW v_rusentiment_stats AS
SELECT label, COUNT(*) AS cnt
FROM rusentiment_posts
GROUP BY label
ORDER BY cnt DESC;

CREATE OR REPLACE VIEW v_helpdesk_stats AS
SELECT sentimen_final,
       COUNT(*)            AS cnt,
       ROUND(AVG(rating_5), 2) AS avg_rating
FROM helpdesk_tickets
GROUP BY sentimen_final
ORDER BY cnt DESC;

CREATE OR REPLACE VIEW v_helpdesk_monthly AS
SELECT DATE_TRUNC('month', request_date) AS month,
       COUNT(*)                          AS total_tickets,
       ROUND(AVG(rating_5), 2)           AS avg_rating,
       SUM(CASE WHEN sentimen_final='negatif' THEN 1 ELSE 0 END) AS negatif_cnt
FROM helpdesk_tickets
GROUP BY 1
ORDER BY 1;