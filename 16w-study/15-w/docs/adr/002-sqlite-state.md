# ADR-002：SQLite WAL 作为共享状态基线

状态：有限接受。

选择 SQLite 是为了零外部依赖、事务化 Run/Checkpoint 和可复现双实例演示。所有键包含租户，写入使用版本 CAS。未选 Mem0，因为本项目不需要跨任务事实记忆；未直接选 PostgreSQL，因为起步容量低。持续 2 writes/s 或 5 GB 时迁移 PostgreSQL，并增加 RLS。

