# Statistical analysis summary

## Paper-level analyses
### Metric: paper_total_score
Friedman chi2 = 7.8710, p = 0.01954, Kendall's W = 0.1711 (n=23, k=3)
Descriptives by condition:
- no_highlights: n=23 mean=6.717391304347826 sd=2.632083891643316
- all_highlights: n=23 mean=7.695652173913044 sd=2.2850260115626226
- contextual_highlights: n=23 mean=7.521739130434782 sd=2.019666156422692
Post-hoc (Wilcoxon, Bonferroni-corrected):
- no_highlights vs all_highlights: stat=23.0000, p=0.01901, p_bonf=0.05702, d_paired=-0.528985951803906
- no_highlights vs contextual_highlights: stat=30.0000, p=0.04674, p_bonf=0.1402, d_paired=-0.4529713628969915
- all_highlights vs contextual_highlights: stat=32.0000, p=0.5768, p_bonf=1, d_paired=0.10775804624090149

### Metric: free_text_score
Friedman chi2 = 3.4915, p = 0.1745, Kendall's W = 0.0759 (n=23, k=3)
Descriptives by condition:
- no_highlights: n=23 mean=3.6956521739130435 sd=1.1845514152353067
- all_highlights: n=23 mean=3.869565217391304 sd=1.3247417527437004
- contextual_highlights: n=23 mean=4.217391304347826 sd=0.9980217586956905
Post-hoc (Wilcoxon, Bonferroni-corrected):
- no_highlights vs all_highlights: stat=56.0000, p=0.4913, p_bonf=1, d_paired=-0.16897599751218176
- no_highlights vs contextual_highlights: stat=20.5000, p=0.06945, p_bonf=0.2083, d_paired=-0.40934808184531796
- all_highlights vs contextual_highlights: stat=20.0000, p=0.1176, p_bonf=0.3527, d_paired=-0.3385842724621209

### Metric: mc_first_attempt_accuracy
Friedman chi2 = 8.0357, p = 0.01799, Kendall's W = 0.1747 (n=23, k=3)
Descriptives by condition:
- no_highlights: n=23 mean=0.6043478260869566 sd=0.35989677922927393
- all_highlights: n=23 mean=0.7652173913043478 sd=0.2604344196847293
- contextual_highlights: n=23 mean=0.6608695652173914 sd=0.3215574746233805
Post-hoc (Wilcoxon, Bonferroni-corrected):
- no_highlights vs all_highlights: stat=24.0000, p=0.04036, p_bonf=0.1211, d_paired=-0.5481577182629344
- no_highlights vs contextual_highlights: stat=38.0000, p=0.361, p_bonf=1, d_paired=-0.25792155466202726
- all_highlights vs contextual_highlights: stat=11.5000, p=0.1018, p_bonf=0.3054, d_paired=0.40934808184531774

### Metric: paper_duration_seconds
Friedman chi2 = 0.6087, p = 0.7376, Kendall's W = 0.0132 (n=23, k=3)
Descriptives by condition:
- no_highlights: n=23 mean=443.4347826086956 sd=200.35192357607252
- all_highlights: n=23 mean=419.60869565217394 sd=215.21463686503438
- contextual_highlights: n=23 mean=429.39130434782606 sd=194.5018670465273
Post-hoc (Wilcoxon, Bonferroni-corrected):
- no_highlights vs all_highlights: stat=111.0000, p=0.4115, p_bonf=1, d_paired=0.16763951681420083
- no_highlights vs contextual_highlights: stat=132.0000, p=0.8552, p_bonf=1, d_paired=0.07178920372531133
- all_highlights vs contextual_highlights: stat=112.5000, p=0.438, p_bonf=1, d_paired=-0.05379542846484301

## Cross-level analyses
### by_contextual_paper
- paper4: n=6 mean=7.833333333333333 sd=1.9407902170679516
- paper5: n=8 mean=9.125 sd=0.6408699444616557
- paper6: n=10 mean=8.0 sd=3.0550504633038935

## Final survey descriptives
- fq_helpful_highlighted_sentences: n=23 mean=3.652173913043478 sd=0.5727680517510726
- fq_confidence_understanding: n=23 mean=3.347826086956522 sd=0.5727680517510725
- fq_identify_important_info: n=23 mean=3.217391304347826 sd=0.7358681786057779
- fq_repeated_known_info: n=23 mean=3.0434782608695654 sd=0.638055345958271
- fq_missing_important_info: n=23 mean=3.0434782608695654 sd=0.7674195764535269
- fq_trust_highlights_key_ideas: n=23 mean=3.782608695652174 sd=0.6712621584563621
- fq_new_vs_repeated: n=23 mean=2.8260869565217392 sd=0.936733876686023
- fq_focus_novel_important_content: n=23 mean=3.260869565217391 sd=0.6887004431501819
- fq_use_in_workflow: n=23 mean=3.5652173913043477 sd=0.5897678246195885
- fq_like_most_least: n=0 mean=None sd=None
- fq_missing_confusing_unnecessary: n=0 mean=None sd=None
### Demographics
- age: {'B': 13, 'A': 5, 'C': 5, 'D': 1}
- role: {'B': 18, 'E': 3, 'C': 2, nan: 1}
- field: {'Computer Science': 6, 'AI/ML': 2, 'computer science': 2, 'Mechatronics': 1, 'Digital Media & CS': 1, nan: 1, 'Artificial Intelliegence': 1, 'CS/ Product Management': 1, 'Machine Learning and Deep Learning': 1, 'CS': 1, 'Data Science': 1, 'Information Experience Design': 1, 'Computer Engineering': 1, 'Computer vision': 1, 'Biomedical Engineering': 1, 'Human Computer Interaction (HCI)': 1, 'Masters of Engineering': 1}
- reading_freq: {'C': 11, 'D': 7, 'B': 5, '3': 1}
- reading_style: {'B': 9, 'D': 7, 'C': 4, 'A': 3, nan: 1}