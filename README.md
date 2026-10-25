# 桌面迁移文件临时传输分支

本分支用于把用户已授权迁移的完整项目状态与原始材料转到桌面，不合并main。

代码和交接文件在main。迁移文件见migration_downloads，逐包SHA256见archive_index.json。

推荐在main仓库运行python scripts/download_migration.py --extract，自动通过Git取包、校验并恢复outputs与data/raw；不会覆盖已改动文件或重新签署历史哈希。
