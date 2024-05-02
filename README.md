# CATE Estimation with Imbalanced Covariates

Conditional average treatment effect (CATE) estimates have been increasingly used
in policy decision-making as they can profile and prioritize individuals that receive
the most benefits from a treatment. In this paper, we study the specific case of an
imbalanced covariate in the data set. We posit that standard parametric and non-
parametric methods lead to disparate performance on the minority and majority
groups, creating bias in CATE estimates. In this paper, we first provide theoreti-
cal derivations for reweighting methods in a parametric setting, which will provide
deeper intuitions about the problem. Then, we propose a repository of tools that
address the issues of imbalanced covariates, including reweighting in causal forests
and data augmentation through generative modelling. We demonstrate the effec-
tiveness of these methods through extensive simulation studies. Finally, we apply
these novel methods to a real-world data set in the case of job training programs.
