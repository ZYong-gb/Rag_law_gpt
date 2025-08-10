-- 初始化数据库脚本

-- 创建数据库（如果不存在）
-- CREATE DATABASE law_gpt;

-- 创建扩展
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- 插入一些示例法律分类
INSERT INTO legal_categories (id, name, description) VALUES 
    (uuid_generate_v4(), '民法', '涉及民事权利义务关系的法律'),
    (uuid_generate_v4(), '刑法', '规定犯罪和刑罚的法律'),
    (uuid_generate_v4(), '行政法', '规范行政管理活动的法律'),
    (uuid_generate_v4(), '商法', '调整商事关系的法律'),
    (uuid_generate_v4(), '劳动法', '调整劳动关系的法律'),
    (uuid_generate_v4(), '知识产权法', '保护知识产权的法律'),
    (uuid_generate_v4(), '环境法', '保护环境的法律法规'),
    (uuid_generate_v4(), '税法', '调整税收关系的法律')
ON CONFLICT (name) DO NOTHING;

-- 创建索引（如果表已存在）
-- 这些索引在模型定义中已经包含，这里作为备份
DO $$
BEGIN
    -- 文档表索引
    IF NOT EXISTS (SELECT 1 FROM pg_indexes WHERE indexname = 'idx_documents_category') THEN
        CREATE INDEX idx_documents_category ON documents(category);
    END IF;
    
    IF NOT EXISTS (SELECT 1 FROM pg_indexes WHERE indexname = 'idx_documents_source') THEN
        CREATE INDEX idx_documents_source ON documents(source);
    END IF;
    
    -- 分块表索引
    IF NOT EXISTS (SELECT 1 FROM pg_indexes WHERE indexname = 'idx_chunks_document_id') THEN
        CREATE INDEX idx_chunks_document_id ON document_chunks(document_id);
    END IF;
    
    -- 查询日志索引
    IF NOT EXISTS (SELECT 1 FROM pg_indexes WHERE indexname = 'idx_query_logs_user_id') THEN
        CREATE INDEX idx_query_logs_user_id ON query_logs(user_id);
    END IF;
    
EXCEPTION
    WHEN undefined_table THEN
        -- 表还不存在，跳过索引创建
        NULL;
END $$;
