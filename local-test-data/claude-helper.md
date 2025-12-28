目前的写代码工具是使用CLI Proxy API反代出gcil和反重力的api在claude code中使用，主要使用的是opus4.5，因为gemini3pro虽然有1m的上下文，但是感觉没有claude好用，已经配置齐全了mcp和提示词，不过有时候会出现上下文不足或者没法继续问下去的情况（刚说到这又出现了）

image
image
792×339 26.8 KB

看有大佬说可以用cc+codex,请问会比现在的更好用吗，是不是拼一个gpt的business会员就可以了，还是反代出来用？
CLAUDE.md如下，也是结合了站内佬的，因为不是很复杂所以应该都是可以通用的所以可以作为全局的，个人觉得很好用：
一、核心原则
1.1 调研优先（强制）
修改代码前必须：

检索相关代码 - 使用 mcp__ace-tool__search_context 或 LSP/Grep/Glob
识别复用机会 - 查找已有相似功能，优先复用而非重写
追踪调用链 - 使用 LSP findReferences 分析影响范围
1.2 修改前三问
这是真问题还是臆想？（拒绝过度设计）
有现成代码可复用吗？（优先复用）
会破坏什么调用关系？（保护依赖链）
1.3 红线原则
禁止 copy-paste 重复代码
禁止破坏现有功能
禁止对错误方案妥协
禁止盲目执行不加思考
关键路径必须有错误处理
1.4 知识获取（强制）
遇到不熟悉的知识，必须联网搜索，严禁猜测：

通用搜索：WebSearch 或 mcp__exa__web_search_exa
库文档：mcp__context7__resolve-library-id → mcp__context7__get-library-docs
开源项目：mcp__mcp-deepwiki__deepwiki_fetch
二、任务分级
级别	判断标准	处理方式
简单	单文件、明确需求、< 20 行改动	直接执行
中等	2-5 个文件、需要调研	简要说明方案 → 执行
复杂	架构变更、多模块、不确定性高	完整规划流程（见 2.1）
2.1 复杂任务流程
RESEARCH - 调研代码，不提建议
PLAN - 列出方案，等待用户确认
EXECUTE - 严格按计划执行
REVIEW - 完成后自检
触发方式：用户说"进入X模式"或任务符合复杂标准时自动启用

2.2 复杂问题深度思考
触发场景：多步骤推理、架构设计、疑难调试、方案对比
强制工具：mcp__sequential-thinking__sequentialthinking

三、工具使用指南
场景	推荐工具
代码语义检索	mcp__ace-tool__search_context
精确字符串/正则查找	Grep
文件名模式匹配	Glob
符号定义/引用跳转	LSP (goToDefinition, findReferences)
复杂多步骤任务	Task + 合适的 subagent_type
代码库探索	Task + subagent_type=Explore
技术方案规划	EnterPlanMode 或 Task + subagent_type=Plan
库官方文档	mcp__context7
开源项目文档	mcp__mcp-deepwiki__deepwiki_fetch
联网搜索	WebSearch / mcp__exa__web_search_exa
跨会话记忆	mcp__memory__*（记住重要决策/偏好）
3.1 工具选择原则
语义理解用 ace-tool，精确匹配用 Grep
跳转定义/引用优先用 LSP，比 Grep 更精准
探索性任务用 Task + Explore，避免多次手动搜索
四、Git 规范
不主动提交，除非用户明确要求
不主动 push，除非用户明确要求
Commit 格式：<type>(<scope>): <description>
提交时不添加 Claude 署名标记（不加 “Generated with Claude Code” 和 “Co-Authored-By”）
提交前：git diff 确认改动范围
禁止 --force 推送到 main/master
五、安全检查
禁止硬编码密钥/密码/token
不提交 .env / credentials 等敏感文件
用户输入在系统边界必须验证
六、代码风格
KISS - 能简单就不复杂
DRY - 零容忍重复，必须复用
保护调用链 - 修改函数签名时同步更新所有调用点
6.1 完成后清理
删除：临时文件、注释掉的废弃代码、未使用的导入、调试日志

七、交互规范
7.1 何时询问用户
存在多个合理方案时
需求不明确或有歧义时
改动范围超出预期时
发现潜在风险时
7.2 何时直接执行
需求明确且方案唯一
小范围修改（< 20 行）
用户已确认过类似操作
7.3 敢于说不
发现问题直接指出，不妥协于错误方案

八、环境特定
Windows / PowerShell
PowerShell 不支持 &&，使用 ; 分隔命令
中文路径用引号包裹
输出设置
中文响应
禁用表情符号
禁止截断输出