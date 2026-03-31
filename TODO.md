# TODO

1. Adapter 插入中间层后，是否解冻之后的LLM层
2. Adapter 是否需要重新构筑，层数更多的MLP或轻量的Transformer
3. 是否考虑改为指令LLM，以适配json形式输出
4. 是否考虑tgt不设为json，而通过纯文本描述