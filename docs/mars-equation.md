# The Auditable MARS Equation

Unlike opaque neural networks or gradient boosting ensembles, `whitebox-dedup` expresses every decision through an auditable Multivariate Adaptive Regression Splines (MARS) piecewise linear equation.

## Mathematical Formulation

The probability of a candidate pair being a duplicate is given by:

$$\text{logit}[P(\text{duplicate})] = \beta_0 + \sum_{i=1}^{k} \beta_i \cdot h_i(x)$$

Where each hinge function $h_i(x)$ is either forward:

$$h_i(x) = \max(0, x - t_i)$$

Or reflected:

$$h_i(x) = \max(0, t_i - x)$$

And the final probability is recovered through the logistic link:

$$P(\text{duplicate}) = \frac{1}{1 + e^{-\text{logit}}}$$

## Spreadsheet Traceability

Every term, knot, and coefficient can be loaded in Microsoft Excel or Google Sheets:

```excel
= 1 / (1 + EXP(-(-102.231 + 562.965*MAX(0, JW - 0.559) - 679.490*MAX(0, JW - 0.806) + 96.830*MAX(0, 0.898 - JW) + 122.623*MAX(0, JW - 0.898))))
```

Field agents and auditors can verify any flagged record with a standard pocket calculator or spreadsheet.
