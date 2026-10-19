-- 0001 (M6a R3): knowledge-base tables. Only in-scope chunks are stored.
-- The embedding column and its HNSW index come in 0002 (R4), once the model fixes the dimension.

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE source_document (
    id          text PRIMARY KEY,           -- "openstax-prealgebra-2e"
    title       text NOT NULL,
    book_url    text NOT NULL,
    repo_url    text NOT NULL,
    commit_sha  text NOT NULL,              -- pinned source commit
    license     text NOT NULL,
    attribution text NOT NULL,              -- shown with every citation (CC BY-NC-SA)
    lang        text NOT NULL
);

CREATE TABLE skill (
    id      text PRIMARY KEY,               -- "frac.add_sub.unlike_den"
    name_vi text NOT NULL,
    name_en text NOT NULL
);

CREATE TABLE chunk (
    id            text PRIMARY KEY,         -- "<module id>:<element id>" (R2-D5)
    source_id     text NOT NULL REFERENCES source_document (id),
    module_id     text NOT NULL,
    section_title text NOT NULL,
    doc_type      text NOT NULL CHECK (doc_type IN
                      ('theory', 'worked_example', 'procedure', 'key_concepts', 'definition')),
    text          text NOT NULL CHECK (text <> ''),
    -- Lexical branch of retrieval (R5): stemmed English tokens, recomputed whenever text changes.
    tsv           tsvector GENERATED ALWAYS AS (to_tsvector('english', text)) STORED
);
CREATE INDEX chunk_source_idx ON chunk (source_id);
CREATE INDEX chunk_tsv_idx ON chunk USING gin (tsv);

CREATE TABLE chunk_skill (
    chunk_id  text NOT NULL REFERENCES chunk (id) ON DELETE CASCADE,
    skill_id  text NOT NULL REFERENCES skill (id),
    tagged_by text NOT NULL CHECK (tagged_by IN ('human', 'llm')),
    PRIMARY KEY (chunk_id, skill_id)
);
CREATE INDEX chunk_skill_skill_idx ON chunk_skill (skill_id);
