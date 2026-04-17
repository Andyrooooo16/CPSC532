# Statistical analysis summary

## Paper-level analyses
### Metric: paper_total_score
Friedman chi2 = 7.6364, p = 0.02197, Kendall's W = 0.1591 (n=24, k=3)
Descriptives by condition:
- no_highlights: n=24 mean=6.75 sd=2.471577563454951
- all_highlights: n=24 mean=7.833333333333333 sd=2.057154359296116
- contextual_highlights: n=24 mean=7.375 sd=2.222659981037214
Post-hoc (Wilcoxon, Bonferroni-corrected):
- no_highlights vs all_highlights: stat=26.0000, p=0.01575, p_bonf=0.04724, d_paired=-0.560787125964255
- no_highlights vs contextual_highlights: stat=45.0000, p=0.1309, p_bonf=0.3926, d_paired=-0.3131388422340623
- all_highlights vs contextual_highlights: stat=33.0000, p=0.3774, p_bonf=1, d_paired=0.2006920410907822

### Metric: free_text_score
Friedman chi2 = 2.5714, p = 0.2765, Kendall's W = 0.0536 (n=24, k=3)
Descriptives by condition:
- no_highlights: n=24 mean=3.75 sd=1.1131623577819225
- all_highlights: n=24 mean=3.9166666666666665 sd=1.212853862900895
- contextual_highlights: n=24 mean=4.125 sd=1.226961606004318
Post-hoc (Wilcoxon, Bonferroni-corrected):
- no_highlights vs all_highlights: stat=64.0000, p=0.5193, p_bonf=1, d_paired=-0.15288552495196328
- no_highlights vs contextual_highlights: stat=33.0000, p=0.2067, p_bonf=0.6201, d_paired=-0.2606019681943551
- all_highlights vs contextual_highlights: stat=31.0000, p=0.2975, p_bonf=0.8926, d_paired=-0.150696120371769

### Metric: mc_first_attempt_accuracy
Friedman chi2 = 8.6333, p = 0.01334, Kendall's W = 0.1799 (n=24, k=3)
Descriptives by condition:
- no_highlights: n=24 mean=0.6 sd=0.3489113503954588
- all_highlights: n=24 mean=0.7833333333333332 sd=0.24257077258017903
- contextual_highlights: n=24 mean=0.65 sd=0.31896571984271693
Post-hoc (Wilcoxon, Bonferroni-corrected):
- no_highlights vs all_highlights: stat=23.5000, p=0.02106, p_bonf=0.06319, d_paired=-0.6106196666318562
- no_highlights vs contextual_highlights: stat=47.0000, p=0.4576, p_bonf=1, d_paired=-0.22458538797354574
- all_highlights vs contextual_highlights: stat=11.5000, p=0.0552, p_bonf=0.1656, d_paired=0.4647175930762186

### Metric: paper_duration_seconds
Friedman chi2 = 3.5579, p = 0.1688, Kendall's W = 0.0741 (n=24, k=3)
Descriptives by condition:
- no_highlights: n=24 mean=531.0416666666666 sd=45.7692503255441
- all_highlights: n=24 mean=535.0416666666666 sd=45.45851349610282
- contextual_highlights: n=24 mean=537.9583333333334 sd=39.23450030894689
Post-hoc (Wilcoxon, Bonferroni-corrected):
- no_highlights vs all_highlights: stat=137.5000, p=0.7209, p_bonf=1, d_paired=-0.06014379790768107
- no_highlights vs contextual_highlights: stat=121.5000, p=0.4153, p_bonf=1, d_paired=-0.13058924614846734
- all_highlights vs contextual_highlights: stat=123.0000, p=0.6482, p_bonf=1, d_paired=-0.04728866246430936

## Cross-level analyses
### by_contextual_paper
- paper4: n=7 mean=7.714285714285714 sd=1.7994708216848745
- paper5: n=8 mean=9.125 sd=0.6408699444616557
- paper6: n=9 mean=8.88888888888889 sd=1.2692955176439846

## Final survey descriptives
- fq_helpful_highlighted_sentences: n=24 mean=3.625 sd=0.5757792451369143
- fq_confidence_understanding: n=24 mean=3.3333333333333335 sd=0.5646597025732799
- fq_identify_important_info: n=24 mean=3.1666666666666665 sd=0.761386987626881
- fq_repeated_known_info: n=24 mean=3.0416666666666665 sd=0.624093545570845
- fq_missing_important_info: n=24 mean=3.0416666666666665 sd=0.7506036218280919
- fq_trust_highlights_key_ideas: n=24 mean=3.7916666666666665 sd=0.6580053301400783
- fq_new_vs_repeated: n=24 mean=2.7916666666666665 sd=0.9315329426211431
- fq_focus_novel_important_content: n=24 mean=3.2916666666666665 sd=0.6902530516863499
- fq_use_in_workflow: n=24 mean=3.5416666666666665 sd=0.5882299658752717
- fq_like_most_least: n=0 mean=None sd=None
- fq_missing_confusing_unnecessary: n=0 mean=None sd=None
### Demographics
- age: {'B': 13, 'A': 6, 'C': 5}
- role: {'B': 19, 'E': 3, 'C': 2}
- field: {'Computer Science': 7, 'AI/ML': 2, 'computer science': 2, 'Mechatronics': 1, 'Digital Media & CS': 1, 'Artificial Intelliegence': 1, 'CS/ Product Management': 1, 'Machine Learning and Deep Learning': 1, 'CS': 1, 'Data Science': 1, 'Information Experience Design': 1, 'Computer Engineering': 1, 'Computer vision': 1, 'Biomedical Engineering': 1, 'Human Computer Interaction (HCI)': 1, 'Masters of Engineering': 1}
- reading_freq: {'C': 11, 'D': 8, 'B': 5}
- reading_style: {'B': 10, 'D': 7, 'C': 4, 'A': 3}