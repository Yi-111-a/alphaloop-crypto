"""ASSET/memory —— 分层记忆库(§3.4)。

ASSET 区代码(策略agent自由读写区，可被agent自我进化，见§3.5)。真正的实现在
engine.py；这里保持为空/薄导出，避免给 evolve 逻辑增加额外的隐式耦合面。
"""
