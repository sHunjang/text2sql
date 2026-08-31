CREATE TABLE agents (
    agent_id     INTEGER PRIMARY KEY,
    agent_name   TEXT NOT NULL,
    agent_type   TEXT,          -- 'planner', 'executor', 'critic'
    created_at   DATETIME
);

CREATE TABLE executions (
    execution_id INTEGER PRIMARY KEY,
    agent_id     INTEGER,
    status       TEXT,          -- 'success', 'failed', 'timeout', 'running'
    started_at   DATETIME,
    ended_at     DATETIME,
    latency_ms   INTEGER,
    error_message TEXT,
    FOREIGN KEY (agent_id) REFERENCES agents(agent_id)
);

CREATE TABLE tool_calls (
    tool_call_id INTEGER PRIMARY KEY,
    execution_id INTEGER,
    tool_name    TEXT,
    input_tokens INTEGER,
    output_tokens INTEGER,
    cost_usd     REAL,
    called_at    DATETIME,
    FOREIGN KEY (execution_id) REFERENCES executions(execution_id)
);
