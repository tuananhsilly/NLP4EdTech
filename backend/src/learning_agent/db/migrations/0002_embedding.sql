-- 0002 (M6a R4): chunk embeddings for dense retrieval, and a record of every ingest run.
-- 1024 is the dimension of BAAI/bge-m3 (config `embedding.dim`, checked at every ingest);
-- a model with another dimension needs a new migration.

ALTER TABLE chunk ADD COLUMN embedding vector(1024);   -- NULL until the ingest embeds the chunk

-- Approximate nearest neighbours by cosine distance (`<=>`). m and ef_construction are
-- pgvector's defaults, written out so the schema states them.
CREATE INDEX chunk_embedding_idx ON chunk USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

CREATE TABLE ingest_run (
    id          bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    started_at  timestamptz NOT NULL,
    finished_at timestamptz NOT NULL DEFAULT now(),
    source_ids  text[] NOT NULL,
    git_commit  text,                -- NULL when not run from a git checkout
    config      jsonb NOT NULL,      -- the whole Settings used by the run
    config_hash text NOT NULL,
    stats       jsonb NOT NULL       -- counts, device, timing, truncated chunks
);
