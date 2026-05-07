# SAM 初步实验结果

## 总体指标

| 指标 | 数值 |
| --- | ---: |
| 数据集来源数 | 1 |
| 查询数量 | 8 |
| 候选文档节点数量 | 80 |
| Gold 支持证据数量 | 16 |
| 纯向量检索证据召回率 | 0.500 |
| 联想图检索证据召回率 | 0.625 |
| 纯向量命中支持证据数 | 8 |
| 联想检索命中支持证据数 | 10 |
| 联想检索新增有效证据数 | 2 |
| 联想检索平均路径长度 | 1.75 |

## 案例分析

### hotpotqa_5a8c7595554299585d9e36b6 (hotpotqa_real)

- 问题：What government position was held by the woman who portrayed Corliss Archer in the film Kiss and Tell?
- 答案：Chief of Protocol
- 支持文档：hotpotqa_5a8c7595554299585d9e36b6_doc_1, hotpotqa_5a8c7595554299585d9e36b6_doc_6
- 纯向量命中支持证据数：1
- 联想检索命中支持证据数：2

| 模式 | 文档 | 是否支持证据 | 分数 | 路径 | 排序原因 |
| --- | --- | --- | ---: | --- | --- |
| 纯向量 | A Kiss for Corliss | 否 | 0.4460 | mem_9f7764d494133890 | 向量相似度=0.412 |
| 纯向量 | Kiss and Tell (1945 film) | 是 | 0.3787 | mem_adb30998bce56b07 | 向量相似度=0.344 |
| 纯向量 | Lord High Treasurer | 否 | 0.2080 | mem_c81341af11a17181 | 向量相似度=0.174 |
| 纯向量 | Meet Corliss Archer | 否 | 0.1977 | mem_7697eee384354e3c | 向量相似度=0.163 |
| 联想检索 | A Kiss for Corliss | 否 | 0.3343 | mem_9f7764d494133890 | 向量种子节点 |
| 联想检索 | Kiss and Tell (1945 film) | 是 | 0.4493 | mem_9f7764d494133890 -> mem_adb30998bce56b07 | 向量种子节点 -> shared_entity(共享实体：Shirley Temple) |
| 联想检索 | Shirley Temple | 是 | 0.2269 | mem_9f7764d494133890 -> mem_370b7d4347cb0f81 | 向量种子节点 -> shared_entity(共享实体：Shirley Temple) |
| 联想检索 | Charles Craft | 否 | 0.1995 | mem_9f7764d494133890 -> mem_b63c9f3723ed1280 | 向量种子节点 -> embedding_similarity(语义相似度达到阈值：0.259) |

### hotpotqa_5a85ea095542994775f606a8 (hotpotqa_real)

- 问题：What science fantasy young adult series, told in first person, has a set of companion books narrating the stories of enslaved worlds and alien species?
- 答案：Animorphs
- 支持文档：hotpotqa_5a85ea095542994775f606a8_doc_2, hotpotqa_5a85ea095542994775f606a8_doc_8
- 纯向量命中支持证据数：1
- 联想检索命中支持证据数：1

| 模式 | 文档 | 是否支持证据 | 分数 | 路径 | 排序原因 |
| --- | --- | --- | ---: | --- | --- |
| 纯向量 | Animorphs | 是 | 0.4476 | mem_72009a704fd2cc0a | 向量相似度=0.413 |
| 纯向量 | Victoria Hanley | 否 | 0.3449 | mem_e19df01c6024a4d1 | 向量相似度=0.310 |
| 纯向量 | Etiquette &amp; Espionage | 否 | 0.3289 | mem_cc891bfccc0690a5 | 向量相似度=0.294 |
| 纯向量 | Left Behind: The Kids | 否 | 0.3130 | mem_162119eecfdf2adf | 向量相似度=0.279 |
| 联想检索 | Animorphs | 是 | 0.3354 | mem_72009a704fd2cc0a | 向量种子节点 |
| 联想检索 | Victoria Hanley | 否 | 0.3879 | mem_72009a704fd2cc0a -> mem_e19df01c6024a4d1 | 向量种子节点 -> keyword_overlap(关键词重叠：adult, young) |
| 联想检索 | Andre Norton Award | 否 | 0.3773 | mem_72009a704fd2cc0a -> mem_6d22c4d45f4f056d | 向量种子节点 -> keyword_overlap(关键词重叠：adult, fantasy, science, young) |
| 联想检索 | Etiquette &amp; Espionage | 否 | 0.3770 | mem_72009a704fd2cc0a -> mem_cc891bfccc0690a5 | 向量种子节点 -> keyword_overlap(关键词重叠：adult, young) |

### hotpotqa_5a8e3ea95542995a26add48d (hotpotqa_real)

- 问题：The director of the romantic comedy "Big Stone Gap" is based in what New York city?
- 答案：Greenwich Village, New York City
- 支持文档：hotpotqa_5a8e3ea95542995a26add48d_doc_3, hotpotqa_5a8e3ea95542995a26add48d_doc_9
- 纯向量命中支持证据数：1
- 联想检索命中支持证据数：2

| 模式 | 文档 | 是否支持证据 | 分数 | 路径 | 排序原因 |
| --- | --- | --- | ---: | --- | --- |
| 纯向量 | Big Stone Gap (film) | 是 | 0.4697 | mem_3ede339a42fdd773 | 向量相似度=0.435 |
| 纯向量 | Just Another Romantic Wrestling Comedy | 否 | 0.4537 | mem_02223dfe801f5e84 | 向量相似度=0.419 |
| 纯向量 | Nola (film) | 否 | 0.4034 | mem_28b5e8e4b3772366 | 向量相似度=0.369 |
| 纯向量 | Clinton, Minnesota | 否 | 0.3647 | mem_b83f975c32430a1e | 向量相似度=0.330 |
| 联想检索 | Big Stone Gap (film) | 是 | 0.3504 | mem_3ede339a42fdd773 | 向量种子节点 |
| 联想检索 | Just Another Romantic Wrestling Comedy | 否 | 0.4028 | mem_3ede339a42fdd773 -> mem_02223dfe801f5e84 | 向量种子节点 -> embedding_similarity(语义相似度达到阈值：0.264) |
| 联想检索 | Clinton, Minnesota | 否 | 0.4014 | mem_3ede339a42fdd773 -> mem_b83f975c32430a1e | 向量种子节点 -> keyword_overlap(关键词重叠：big, stone) |
| 联想检索 | Adriana Trigiani | 是 | 0.3744 | mem_3ede339a42fdd773 -> mem_36dabff8cb904650 | 向量种子节点 -> shared_entity(共享实体：Adriana Trigiani) |

### hotpotqa_5a87ab905542996e4f3088c1 (hotpotqa_real)

- 问题：The arena where the Lewiston Maineiacs played their home games can seat how many people?
- 答案：3,677 seated
- 支持文档：hotpotqa_5a87ab905542996e4f3088c1_doc_5, hotpotqa_5a87ab905542996e4f3088c1_doc_7
- 纯向量命中支持证据数：1
- 联想检索命中支持证据数：1

| 模式 | 文档 | 是否支持证据 | 分数 | 路径 | 排序原因 |
| --- | --- | --- | ---: | --- | --- |
| 纯向量 | Billings Bulls | 否 | 0.4101 | mem_4ceae82d95b7f531 | 向量相似度=0.376 |
| 纯向量 | Dwyer Arena | 否 | 0.3948 | mem_a9e78f51bbf34d88 | 向量相似度=0.360 |
| 纯向量 | Lewiston Maineiacs | 是 | 0.3444 | mem_6b885c37ddbd517b | 向量相似度=0.310 |
| 纯向量 | Case Gym | 否 | 0.2908 | mem_f74767a19e693588 | 向量相似度=0.256 |
| 联想检索 | Billings Bulls | 否 | 0.3099 | mem_4ceae82d95b7f531 | 向量种子节点 |
| 联想检索 | Dwyer Arena | 否 | 0.4411 | mem_4ceae82d95b7f531 -> mem_a9e78f51bbf34d88 | 向量种子节点 -> keyword_overlap(关键词重叠：arena, hockey, ice) |
| 联想检索 | Lewiston Maineiacs | 是 | 0.4068 | mem_4ceae82d95b7f531 -> mem_6b885c37ddbd517b | 向量种子节点 -> keyword_overlap(关键词重叠：hockey, ice, junior) |
| 联想检索 | Case Gym | 否 | 0.3704 | mem_4ceae82d95b7f531 -> mem_f74767a19e693588 | 向量种子节点 -> keyword_overlap(关键词重叠：arena, games, played) |

### hotpotqa_5a7bbb64554299042af8f7cc (hotpotqa_real)

- 问题：Who is older, Annie Morton or Terry Richardson?
- 答案：Terry Richardson
- 支持文档：hotpotqa_5a7bbb64554299042af8f7cc_doc_0, hotpotqa_5a7bbb64554299042af8f7cc_doc_2
- 纯向量命中支持证据数：2
- 联想检索命中支持证据数：2

| 模式 | 文档 | 是否支持证据 | 分数 | 路径 | 排序原因 |
| --- | --- | --- | ---: | --- | --- |
| 纯向量 | Annie Morton | 是 | 0.4222 | mem_7f496efdb1d7f3b5 | 向量相似度=0.388 |
| 纯向量 | Lady Gaga x Terry Richardson | 否 | 0.3835 | mem_93518e8c9d6185ca | 向量相似度=0.349 |
| 纯向量 | Terry Richardson | 是 | 0.3255 | mem_35a69d78ac155552 | 向量相似度=0.291 |
| 纯向量 | Gumbo (PJ Morton album) | 否 | 0.3144 | mem_6d02e64f0e07d876 | 向量相似度=0.280 |
| 联想检索 | Annie Morton | 是 | 0.3181 | mem_7f496efdb1d7f3b5 | 向量种子节点 |
| 联想检索 | Lady Gaga x Terry Richardson | 否 | 0.4526 | mem_7f496efdb1d7f3b5 -> mem_93518e8c9d6185ca | 向量种子节点 -> shared_entity(共享实体：Terry Richardson) |
| 联想检索 | Terry Richardson | 是 | 0.4132 | mem_7f496efdb1d7f3b5 -> mem_35a69d78ac155552 | 向量种子节点 -> shared_entity(共享实体：Terry Richardson) |
| 联想检索 | Kenton Richardson | 否 | 0.3854 | mem_7f496efdb1d7f3b5 -> mem_506385d5a587512d | 向量种子节点 -> shared_entity(共享实体：Terry Richardson) |

### hotpotqa_5a7166395542994082a3e814 (hotpotqa_real)

- 问题：What is the name of the fight song of the university whose main campus is in Lawrence, Kansas and whose branch campuses are in the Kansas City metropolitan area?
- 答案：Kansas Song
- 支持文档：hotpotqa_5a7166395542994082a3e814_doc_4, hotpotqa_5a7166395542994082a3e814_doc_9
- 纯向量命中支持证据数：2
- 联想检索命中支持证据数：2

| 模式 | 文档 | 是否支持证据 | 分数 | 路径 | 排序原因 |
| --- | --- | --- | ---: | --- | --- |
| 纯向量 | University of Kansas | 是 | 0.7305 | mem_a1b72eb596ed5980 | 向量相似度=0.696 |
| 纯向量 | Kansas Song | 是 | 0.6379 | mem_eab44424e87bdaff | 向量相似度=0.604 |
| 纯向量 | Kansas City metropolitan area | 否 | 0.6215 | mem_1fcc9458ae962c86 | 向量相似度=0.587 |
| 纯向量 | Kansas City jazz | 否 | 0.5784 | mem_66cd87315fc13282 | 向量相似度=0.544 |
| 联想检索 | University of Kansas | 是 | 0.5278 | mem_a1b72eb596ed5980 | 向量种子节点 |
| 联想检索 | Kansas Song | 是 | 0.6256 | mem_a1b72eb596ed5980 -> mem_eab44424e87bdaff | 向量种子节点 -> shared_entity(共享实体：University of Kansas) |
| 联想检索 | Kansas City metropolitan area | 否 | 0.6144 | mem_a1b72eb596ed5980 -> mem_1fcc9458ae962c86 | 向量种子节点 -> shared_entity(共享实体：Kansas City metropolitan area) |
| 联想检索 | Kansas City jazz | 否 | 0.5851 | mem_a1b72eb596ed5980 -> mem_66cd87315fc13282 | 向量种子节点 -> shared_entity(共享实体：Kansas City metropolitan area) |

### hotpotqa_5a877e5d5542993e715abf7d (hotpotqa_real)

- 问题：What screenwriter with credits for "Evolution" co-wrote a film starring Nicolas Cage and Téa Leoni?
- 答案：David Weissman
- 支持文档：hotpotqa_5a877e5d5542993e715abf7d_doc_3, hotpotqa_5a877e5d5542993e715abf7d_doc_9
- 纯向量命中支持证据数：0
- 联想检索命中支持证据数：0

| 模式 | 文档 | 是否支持证据 | 分数 | 路径 | 排序原因 |
| --- | --- | --- | ---: | --- | --- |
| 纯向量 | Deadfall (1993 film) | 否 | 0.3904 | mem_3205f471d2bf81cb | 向量相似度=0.356 |
| 纯向量 | Time to Kill (1989 film) | 否 | 0.3602 | mem_904b0758d46c0f41 | 向量相似度=0.326 |
| 纯向量 | It Could Happen to You (1994 film) | 否 | 0.3490 | mem_c8b77d88da839a54 | 向量相似度=0.315 |
| 纯向量 | Gone in 60 Seconds (2000 film) | 否 | 0.3383 | mem_b2d0baffaf5aaba2 | 向量相似度=0.304 |
| 联想检索 | Deadfall (1993 film) | 否 | 0.2965 | mem_3205f471d2bf81cb | 向量种子节点 |
| 联想检索 | It Could Happen to You (1994 film) | 否 | 0.3490 | mem_3205f471d2bf81cb -> mem_c8b77d88da839a54 | 向量种子节点 -> embedding_similarity(语义相似度达到阈值：0.336) |
| 联想检索 | Gone in 60 Seconds (2000 film) | 否 | 0.3357 | mem_3205f471d2bf81cb -> mem_b2d0baffaf5aaba2 | 向量种子节点 -> embedding_similarity(语义相似度达到阈值：0.311) |
| 联想检索 | Time to Kill (1989 film) | 否 | 0.3351 | mem_3205f471d2bf81cb -> mem_904b0758d46c0f41 | 向量种子节点 -> embedding_similarity(语义相似度达到阈值：0.246) |

### hotpotqa_5ab6d09255429954757d337d (hotpotqa_real)

- 问题：The football manager who recruited David Beckham managed Manchester United during what timeframe?
- 答案：from 1986 to 2013
- 支持文档：hotpotqa_5ab6d09255429954757d337d_doc_5, hotpotqa_5ab6d09255429954757d337d_doc_9
- 纯向量命中支持证据数：0
- 联想检索命中支持证据数：0

| 模式 | 文档 | 是否支持证据 | 分数 | 路径 | 排序原因 |
| --- | --- | --- | ---: | --- | --- |
| 纯向量 | David Beckham's Soccer USA | 否 | 0.3909 | mem_304ea0fca865a016 | 向量相似度=0.356 |
| 纯向量 | David Beckham Academy | 否 | 0.3539 | mem_3190e94565b6750e | 向量相似度=0.320 |
| 纯向量 | Ernest Mangnall | 否 | 0.3055 | mem_86af338e81fe6f91 | 向量相似度=0.271 |
| 纯向量 | The Class of '92 | 否 | 0.3001 | mem_eda02b77d7ecea3b | 向量相似度=0.266 |
| 联想检索 | David Beckham's Soccer USA | 否 | 0.2968 | mem_304ea0fca865a016 | 向量种子节点 |
| 联想检索 | David Beckham Academy | 否 | 0.3941 | mem_304ea0fca865a016 -> mem_3190e94565b6750e | 向量种子节点 -> keyword_overlap(关键词重叠：beckham, david) |
| 联想检索 | The Class of '92 | 否 | 0.2817 | mem_304ea0fca865a016 -> mem_eda02b77d7ecea3b | 向量种子节点 -> embedding_similarity(语义相似度达到阈值：0.194) |
| 联想检索 | Ernest Mangnall | 否 | 0.2579 | mem_304ea0fca865a016 -> mem_86af338e81fe6f91 | 向量种子节点 -> context_cooccurrence(同一公开多跳问答样本中的候选上下文，保留跨文档推理的候选关系) |
