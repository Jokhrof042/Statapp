from flask import Flask, render_template, request, jsonify
import statistics
import math


app = Flask(__name__)

def parse_numbers(s):
    s = s.replace(',', ' ')
    return [float(x) for x in s.split() if x.strip()]

def safe(val, digits=6):
    if isinstance(val, float):
        return round(val, digits)
    return val

# ── 1. Descriptive Statistics (AST 101) ──────────────────────────────────────
def calc_descriptive(nums):
    n = len(nums)
    if n == 0: return {"error": "No numbers provided."}
    mean = statistics.mean(nums)
    median = statistics.median(nums)
    try: mode = statistics.mode(nums)
    except: mode = "No unique mode"
    sorted_nums = sorted(nums)
    q1 = sorted_nums[n // 4]
    q3 = sorted_nums[(3 * n) // 4]
    pop_var = statistics.pvariance(nums) if n >= 2 else 0
    samp_var = statistics.variance(nums) if n >= 2 else 0
    pop_std = statistics.pstdev(nums) if n >= 2 else 0
    samp_std = statistics.stdev(nums) if n >= 2 else 0
    cv = (samp_std / mean * 100) if mean != 0 else 0
    if pop_std > 0:
        skewness = sum((x - mean) ** 3 for x in nums) / (n * pop_std ** 3)
        kurtosis = sum((x - mean) ** 4 for x in nums) / (n * pop_std ** 4) - 3
    else:
        skewness = kurtosis = 0
    return {
        "count_n": n, "sum": safe(sum(nums)), "mean": safe(mean),
        "median": safe(median), "mode": mode,
        "minimum": safe(min(nums)), "maximum": safe(max(nums)),
        "range": safe(max(nums) - min(nums)),
        "Q1": safe(q1), "Q3": safe(q3), "IQR": safe(q3 - q1),
        "pop_variance": safe(pop_var), "sample_variance": safe(samp_var),
        "pop_std_dev": safe(pop_std), "sample_std_dev": safe(samp_std),
        "coeff_variation_%": safe(cv, 4),
        "skewness": safe(skewness), "excess_kurtosis": safe(kurtosis),
    }

# ── 2. Permutation & Combination (AST 102) ───────────────────────────────────
def calc_permcomb(n, r):
    n, r = int(n), int(r)
    if r > n: return {"error": "r cannot be greater than n."}
    perm = math.factorial(n) // math.factorial(n - r)
    comb = math.factorial(n) // (math.factorial(r) * math.factorial(n - r))
    return {"n": n, "r": r, "permutation_nPr": perm, "combination_nCr": comb}

# ── 3. Binomial Distribution (AST 102) ───────────────────────────────────────
def calc_binomial(n, k, p):
    n, k = int(n), int(k)
    p = float(p)
    if not 0 <= p <= 1: return {"error": "p must be between 0 and 1."}
    if k > n: return {"error": "k cannot exceed n."}
    comb = math.factorial(n) // (math.factorial(k) * math.factorial(n - k))
    prob = comb * (p ** k) * ((1 - p) ** (n - k))
    mean = n * p; var = n * p * (1 - p)
    return {
        "P(X=k)": safe(prob), "mean_np": safe(mean),
        "variance_npq": safe(var), "std_dev": safe(math.sqrt(var)),
        "formula": f"C({n},{k}) x {p}^{k} x {round(1-p,4)}^{n-k}"
    }

# ── 4. Normal Distribution (AST 102, AST 201) ────────────────────────────────
def calc_normal(mean, std, x1, x2=None, tail="less"):
    mean, std = float(mean), float(std)
    x1 = float(x1)
    def phi(x): return 0.5 * (1 + math.erf((x - mean) / (std * math.sqrt(2))))
    if tail == "less":
        prob = phi(x1); z = (x1 - mean) / std
        return {"P(X_less_x1)": safe(prob), "z_score": safe(z)}
    elif tail == "greater":
        prob = 1 - phi(x1); z = (x1 - mean) / std
        return {"P(X_greater_x1)": safe(prob), "z_score": safe(z)}
    elif tail == "between" and x2 is not None:
        x2 = float(x2)
        prob = phi(x2) - phi(x1)
        return {"P(x1_less_X_less_x2)": safe(prob), "z1": safe((x1-mean)/std), "z2": safe((x2-mean)/std)}
    return {"error": "Invalid parameters."}

# ── 5. Regression (AST 303) ──────────────────────────────────────────────────
def calc_regression(x_vals, y_vals):
    n = len(x_vals)
    if n != len(y_vals) or n < 2: return {"error": "X and Y must have equal count (>=2)."}
    sx = sum(x_vals); sy = sum(y_vals)
    sxy = sum(x*y for x,y in zip(x_vals, y_vals))
    sx2 = sum(x**2 for x in x_vals); sy2 = sum(y**2 for y in y_vals)
    denom = n*sx2 - sx**2
    if denom == 0: return {"error": "All X values identical."}
    b1 = (n*sxy - sx*sy) / denom
    b0 = (sy - b1*sx) / n
    num_r = n*sxy - sx*sy
    den_r = math.sqrt((n*sx2 - sx**2)*(n*sy2 - sy**2))
    r = num_r/den_r if den_r != 0 else 0
    r2 = r**2
    y_pred = [b1*x+b0 for x in x_vals]
    sse = sum((y-yp)**2 for y,yp in zip(y_vals, y_pred))
    se = math.sqrt(sse/(n-2)) if n > 2 else 0
    return {
        "slope_b1": safe(b1), "intercept_b0": safe(b0),
        "equation": f"y_hat = {round(b1,4)}x + {round(b0,4)}",
        "pearson_r": safe(r), "r_squared": safe(r2),
        "std_error_estimate": safe(se), "n": n
    }

# ── 6. Z-Test (AST 203) ──────────────────────────────────────────────────────
def calc_ztest(sample_mean, pop_mean, pop_std, n):
    sample_mean, pop_mean, pop_std, n = float(sample_mean), float(pop_mean), float(pop_std), int(n)
    se = pop_std / math.sqrt(n)
    z = (sample_mean - pop_mean) / se
    p = 2*(1 - 0.5*(1 + math.erf(abs(z)/math.sqrt(2))))
    return {
        "z_statistic": safe(z), "standard_error": safe(se),
        "p_value_two_tailed": safe(p),
        "reject_H0_at_5%": "Yes" if p < 0.05 else "No",
        "reject_H0_at_1%": "Yes" if p < 0.01 else "No",
    }

# ── 7. t-Test helpers ────────────────────────────────────────────────────────
def _betainc(a, b, x, it=200):
    if x <= 0: return 0.0
    if x >= 1: return 1.0
    lbeta = math.lgamma(a)+math.lgamma(b)-math.lgamma(a+b)
    front = math.exp(math.log(x)*a+math.log(1-x)*b-lbeta)/a
    f=1.0; c=1.0; d=1.0-(a+b)*x/(a+1)
    if abs(d)<1e-30: d=1e-30
    d=1/d; f=d
    for m in range(1,it+1):
        for pm in [1,-1]:
            num = m*(b-m)*x/((a+2*m-1)*(a+2*m)) if pm==1 else -(a+m)*(a+b+m)*x/((a+2*m)*(a+2*m+1))
            d=1+num*d; c=1+num/c
            if abs(d)<1e-30: d=1e-30
            if abs(c)<1e-30: c=1e-30
            d=1/d; f*=c*d
    return min(front*f,1.0)

def calc_ttest(nums, pop_mean):
    n = len(nums)
    if n < 2: return {"error": "Need at least 2 observations."}
    pop_mean = float(pop_mean)
    xbar = statistics.mean(nums); s = statistics.stdev(nums)
    se = s / math.sqrt(n); t = (xbar - pop_mean) / se
    df = n - 1; x = df/(df + t*t)
    p = min(_betainc(df/2, 0.5, x), 1.0)
    return {
        "sample_mean": safe(xbar), "sample_std": safe(s),
        "standard_error": safe(se), "t_statistic": safe(t),
        "degrees_of_freedom": df, "p_value_approx_two_tailed": safe(p),
        "reject_H0_at_5%": "Yes" if p < 0.05 else "No",
    }

# ── 8. Confidence Interval (AST 203) ─────────────────────────────────────────
def calc_ci(nums, conf):
    n = len(nums)
    if n < 2: return {"error": "Need at least 2 observations."}
    mean = statistics.mean(nums); se = statistics.stdev(nums) / math.sqrt(n)
    z = {90: 1.645, 95: 1.96, 99: 2.576}.get(int(conf), 1.96)
    margin = z * se
    return {
        "sample_mean": safe(mean), "std_error": safe(se),
        "confidence_level_%": conf, "z_critical": z,
        "margin_of_error": safe(margin),
        "lower_bound": safe(mean - margin), "upper_bound": safe(mean + margin),
        "interval": f"({round(mean-margin,4)},  {round(mean+margin,4)})",
    }

# ── 9. Index Numbers (AST 106) ───────────────────────────────────────────────
def calc_index(bp, cp, qty=None):
    n = len(bp)
    if n != len(cp): return {"error": "Base and current price lists must be equal length."}
    relatives = [c/b*100 if b!=0 else 0 for b,c in zip(bp,cp)]
    result = {
        "price_relatives": [safe(r,2) for r in relatives],
        "simple_avg_price_index": safe(sum(relatives)/n, 4),
    }
    if qty and len(qty) == n:
        lasp = sum(c*q for c,q in zip(cp,qty)) / sum(b*q for b,q in zip(bp,qty))*100
        result["laspeyres_index"] = safe(lasp,4)
        result["inflation_%"] = safe(lasp-100,4)
    return result

# ── 10. Life Table (AST 305) ─────────────────────────────────────────────────
def calc_life_table(lx_list):
    rows = []
    for i, lx in enumerate(lx_list):
        if i < len(lx_list)-1:
            dx = lx_list[i]-lx_list[i+1]; qx = dx/lx if lx>0 else 0; px = 1-qx
        else:
            dx = lx; qx = 1.0; px = 0.0
        rows.append({"age_x": i, "lx": int(lx), "dx": int(dx), "qx": round(qx,6), "px": round(px,6)})
    return {"life_table": rows}

# ── 11. Correlation Matrix (AST 307) ─────────────────────────────────────────
def calc_corrmatrix(datasets):
    k = len(datasets); n = len(datasets[0])
    for d in datasets:
        if len(d) != n: return {"error": "All variables must have equal observations."}
    means = [sum(d)/n for d in datasets]
    matrix = []
    for i in range(k):
        row = []
        for j in range(k):
            xi=[x-means[i] for x in datasets[i]]; xj=[x-means[j] for x in datasets[j]]
            num=sum(a*b for a,b in zip(xi,xj))
            den=math.sqrt(sum(a**2 for a in xi)*sum(b**2 for b in xj))
            row.append(safe(num/den if den!=0 else 1.0, 4))
        matrix.append(row)
    return {"correlation_matrix": matrix, "variables": k, "n": n}

# ── 12. Markov Chain (AST 401) ───────────────────────────────────────────────
def calc_markov(matrix_str, steps, init_str):
    lines = [l.strip() for l in matrix_str.strip().split('\n') if l.strip()]
    P = [parse_numbers(l) for l in lines]
    n = len(P)
    for row in P:
        if len(row) != n: return {"error": "Transition matrix must be square (n x n)."}
        if abs(sum(row) - 1.0) > 0.01: return {"error": f"Each row must sum to 1. Got {round(sum(row),4)}."}
    init = parse_numbers(init_str)
    if len(init) != n: return {"error": f"Initial state vector must have {n} values."}
    if abs(sum(init) - 1.0) > 0.01: return {"error": "Initial state probabilities must sum to 1."}

    steps = int(steps)
    state = list(init)
    history = [list(state)]
    for _ in range(steps):
        new_state = []
        for j in range(n):
            new_state.append(sum(state[i]*P[i][j] for i in range(n)))
        state = new_state
        history.append([round(x,6) for x in state])

    # Steady state (power iteration)
    ss = list(init)
    for _ in range(1000):
        new_ss = []
        for j in range(n):
            new_ss.append(sum(ss[i]*P[i][j] for i in range(n)))
        ss = new_ss

    result = {
        "states_n": n,
        "steps": steps,
        "steady_state": [round(x,6) for x in ss],
    }
    for i, h in enumerate(history):
        result[f"state_after_{i}_steps"] = h
    return {"markov": result, "history": history, "steady_state": [round(x,6) for x in ss], "n": n}

# ── 13. Time Series — Moving Average & Autocorrelation (AST 407) ─────────────
def calc_timeseries(nums, lag, ma_order):
    n = len(nums); lag = int(lag); ma_order = int(ma_order)
    if n < 4: return {"error": "Need at least 4 data points."}
    mean = sum(nums)/n
    # Autocorrelation at given lag
    if lag >= n: return {"error": "Lag must be less than number of observations."}
    var = sum((x-mean)**2 for x in nums)/n
    if var == 0: return {"error": "All values are identical — variance is zero."}
    acf = sum((nums[t]-mean)*(nums[t+lag]-mean) for t in range(n-lag)) / (n*var)
    # Moving average
    ma = []
    if ma_order >= 1 and ma_order <= n:
        for i in range(n - ma_order + 1):
            ma.append(round(sum(nums[i:i+ma_order])/ma_order, 4))
    # Simple trend (linear regression on time)
    t_vals = list(range(1, n+1))
    st = sum(t_vals); sy = sum(nums); st2 = sum(t**2 for t in t_vals)
    sty = sum(t*y for t,y in zip(t_vals,nums))
    denom = n*st2 - st**2
    trend_slope = (n*sty - st*sy)/denom if denom!=0 else 0
    trend_intercept = (sy - trend_slope*st)/n
    return {
        "n_observations": n,
        "mean": safe(mean),
        "std_dev": safe(math.sqrt(var)),
        f"autocorrelation_lag_{lag}": safe(acf),
        f"moving_average_order_{ma_order}": ma,
        "trend_slope": safe(trend_slope),
        "trend_intercept": safe(trend_intercept),
        "trend_equation": f"y = {round(trend_slope,4)}t + {round(trend_intercept,4)}",
    }

# ── 14. Logistic Regression (AST 408) ────────────────────────────────────────
def calc_logistic(x_vals, y_vals, iterations=1000, lr=0.1):
    n = len(x_vals)
    if n != len(y_vals) or n < 4: return {"error": "Need equal X and Y with at least 4 points."}
    for y in y_vals:
        if y not in (0, 1): return {"error": "Y values must be binary (0 or 1)."}
    # Gradient descent
    b0 = 0.0; b1 = 0.0
    def sigmoid(z): return 1/(1+math.exp(-max(-500,min(500,z))))
    for _ in range(int(iterations)):
        db0 = db1 = 0
        for x, y in zip(x_vals, y_vals):
            pred = sigmoid(b0 + b1*x)
            err = pred - y
            db0 += err; db1 += err*x
        b0 -= lr*db0/n; b1 -= lr*db1/n
    # Log-likelihood
    ll = sum(y*math.log(max(1e-15,sigmoid(b0+b1*x)))+(1-y)*math.log(max(1e-15,1-sigmoid(b0+b1*x))) for x,y in zip(x_vals,y_vals))
    # Predictions
    preds = [1 if sigmoid(b0+b1*x)>=0.5 else 0 for x in x_vals]
    accuracy = sum(p==y for p,y in zip(preds,y_vals))/n*100
    odds_ratio = math.exp(b1)
    return {
        "intercept_b0": safe(b0), "coefficient_b1": safe(b1),
        "odds_ratio": safe(odds_ratio),
        "log_likelihood": safe(ll),
        "accuracy_%": safe(accuracy, 2),
        "equation": f"log(p/1-p) = {round(b0,4)} + {round(b1,4)}x",
        "interpretation": f"1 unit increase in X multiplies odds by {round(odds_ratio,4)}",
    }

# ── 15. PCA (AST 403) ────────────────────────────────────────────────────────
def calc_pca(datasets):
    k = len(datasets); n = len(datasets[0])
    for d in datasets:
        if len(d) != n: return {"error": "All variables must have equal observations."}
    # Center data
    means = [sum(d)/n for d in datasets]
    centered = [[datasets[i][j]-means[i] for j in range(n)] for i in range(k)]
    # Covariance matrix
    cov = []
    for i in range(k):
        row = []
        for j in range(k):
            c = sum(centered[i][t]*centered[j][t] for t in range(n))/(n-1)
            row.append(safe(c,4))
        cov.append(row)
    # Variances (diagonal = variance of each variable)
    variances = [cov[i][i] for i in range(k)]
    total_var = sum(variances)
    prop_var = [safe(v/total_var*100,2) if total_var>0 else 0 for v in variances]
    return {
        "covariance_matrix": cov,
        "variable_variances": variances,
        "proportion_of_variance_%": prop_var,
        "total_variance": safe(total_var),
        "n_variables": k,
        "n_observations": n,
        "note": "Diagonal of cov matrix = variance per variable. Largest variance drives PC1.",
    }

# ── 16. Survival Analysis — Kaplan-Meier (AST 405) ──────────────────────────
def calc_survival(times_str, events_str):
    times = parse_numbers(times_str)
    events = [int(x) for x in parse_numbers(events_str)]
    if len(times) != len(events): return {"error": "Times and events must have equal length."}
    n = len(times)

    # Sort by time (censored after events at same time)
    paired = sorted(zip(times, events), key=lambda x: (x[0], x[1]))

    # Kaplan-Meier: Ŝ(t+) = ∏ (nj - dj) / nj  for all j: tj <= t
    # Build ordered event list
    rows = []
    S = 1.0
    at_risk = n
    i = 0
    all_times = [p[0] for p in paired]
    all_events = [p[1] for p in paired]

    processed = set()
    for idx, (t, e) in enumerate(paired):
        if t in processed:
            continue
        processed.add(t)
        # Count events (d) and total at this time point
        nj = sum(1 for ti in all_times if ti >= t)   # at risk at time t
        dj = sum(1 for ti, ei in paired if ti == t and ei == 1)  # deaths at t
        cj = sum(1 for ti, ei in paired if ti == t and ei == 0)  # censored at t
        label = f"{t}+" if cj > 0 and dj == 0 else str(t)

        if dj > 0:
            factor = (nj - dj) / nj
            S = S * factor
            rows.append({
                "time": label,
                "at_risk_nj": nj,
                "deaths_dj": dj,
                "censored": cj,
                "factor_(nj-dj)/nj": round((nj - dj) / nj, 6),
                "S_hat_t": round(S, 6),
            })
        else:
            # Censored only — S does not change, show row
            rows.append({
                "time": label,
                "at_risk_nj": nj,
                "deaths_dj": 0,
                "censored": cj,
                "factor_(nj-dj)/nj": "—",
                "S_hat_t": round(S, 6),
            })

    # Median survival: smallest t where S(t) <= 0.5
    median_surv = next((r["time"] for r in rows if isinstance(r["S_hat_t"], float) and r["S_hat_t"] <= 0.5), "Not reached")

    return {
        "km_table": rows,
        "median_survival": median_surv,
        "n_total": n,
        "total_events": sum(events),
        "total_censored": n - sum(events),
        "formula": "S_hat(t+) = product of (nj - dj)/nj for all tj <= t",
    }

# ── 17. OLS Regression with Diagnostics (AST 404) ────────────────────────────
def calc_ols(x_vals, y_vals):
    n = len(x_vals)
    if n != len(y_vals) or n < 3: return {"error": "Need equal X and Y with at least 3 points."}
    sx=sum(x_vals); sy=sum(y_vals)
    sxy=sum(x*y for x,y in zip(x_vals,y_vals))
    sx2=sum(x**2 for x in x_vals); sy2=sum(y**2 for y in y_vals)
    denom=n*sx2-sx**2
    if denom==0: return {"error": "All X values identical."}
    b1=(n*sxy-sx*sy)/denom; b0=(sy-b1*sx)/n
    y_pred=[b1*x+b0 for x in x_vals]
    y_mean=sy/n
    sse=sum((y-yp)**2 for y,yp in zip(y_vals,y_pred))
    sst=sum((y-y_mean)**2 for y in y_vals)
    ssr=sst-sse
    r2=1-sse/sst if sst!=0 else 0
    adj_r2=1-(1-r2)*(n-1)/(n-2) if n>2 else 0
    mse=sse/(n-2) if n>2 else 0
    se_b1=math.sqrt(mse/sum((x-sx/n)**2 for x in x_vals)) if mse>0 else 0
    se_b0=math.sqrt(mse*(1/n+((sx/n)**2)/sum((x-sx/n)**2 for x in x_vals))) if mse>0 else 0
    t_b1=b1/se_b1 if se_b1>0 else 0
    t_b0=b0/se_b0 if se_b0>0 else 0
    f_stat=ssr/(sse/(n-2)) if sse>0 else 0
    aic=n*math.log(sse/n)+4 if sse>0 else 0
    return {
        "intercept_b0": safe(b0), "slope_b1": safe(b1),
        "equation": f"y = {round(b0,4)} + {round(b1,4)}x",
        "R_squared": safe(r2), "Adj_R_squared": safe(adj_r2),
        "SSE": safe(sse), "SSR": safe(ssr), "SST": safe(sst),
        "MSE": safe(mse), "RMSE": safe(math.sqrt(mse)),
        "SE_b0": safe(se_b0), "SE_b1": safe(se_b1),
        "t_stat_b0": safe(t_b0), "t_stat_b1": safe(t_b1),
        "F_statistic": safe(f_stat), "AIC": safe(aic),
        "n": n,
    }

# ── Routes ───────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/calculate", methods=["POST"])
def calculate():
    data = request.get_json()
    t = data.get("type")
    result = {}
    try:
        if t == "descriptive":
            result = calc_descriptive(parse_numbers(data["numbers"]))
        elif t == "permcomb":
            result = calc_permcomb(data["n"], data["r"])
        elif t == "binomial":
            result = calc_binomial(data["n"], data["k"], data["p"])
        elif t == "normal":
            result = calc_normal(data["mean"], data["std"], data["x1"], data.get("x2"), data.get("tail","less"))
        elif t == "regression":
            result = calc_regression(parse_numbers(data["x"]), parse_numbers(data["y"]))
        elif t == "ztest":
            result = calc_ztest(data["sample_mean"], data["pop_mean"], data["pop_std"], data["n"])
        elif t == "ttest":
            result = calc_ttest(parse_numbers(data["numbers"]), data["pop_mean"])
        elif t == "ci":
            result = calc_ci(parse_numbers(data["numbers"]), int(data["confidence"]))
        elif t == "index":
            qty = parse_numbers(data.get("quantities","")) if data.get("quantities","").strip() else None
            result = calc_index(parse_numbers(data["base_prices"]), parse_numbers(data["current_prices"]), qty)
        elif t == "lifetable":
            result = calc_life_table([int(x) for x in parse_numbers(data["lx"])])
        elif t == "corrmatrix":
            lines = [l.strip() for l in data.get("datasets","").strip().split('\n') if l.strip()]
            result = calc_corrmatrix([parse_numbers(l) for l in lines])
        elif t == "markov":
            result = calc_markov(data["matrix"], data["steps"], data["init"])
        elif t == "timeseries":
            result = calc_timeseries(parse_numbers(data["numbers"]), data.get("lag",1), data.get("ma_order",3))
        elif t == "logistic":
            result = calc_logistic(parse_numbers(data["x"]), parse_numbers(data["y"]))
        elif t == "pca":
            lines = [l.strip() for l in data.get("datasets","").strip().split('\n') if l.strip()]
            result = calc_pca([parse_numbers(l) for l in lines])
        elif t == "survival":
            result = calc_survival(data["times"], data["events"])
        elif t == "ols":
            result = calc_ols(parse_numbers(data["x"]), parse_numbers(data["y"]))
    except Exception as e:
        result = {"error": str(e)}
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True)

# ── Plot Route ────────────────────────────────────────────────────────────────
PLOT_THEME = dict(
    paper_bgcolor='#0b0e13', plot_bgcolor='#12161e',
    font=dict(color='#e8edf5', family='Outfit, sans-serif'),
    margin=dict(l=50, r=30, t=50, b=50),
)

def fig_json(fig):
    import plotly.graph_objects as go
    fig.update_layout(**PLOT_THEME)
    fig.update_xaxes(gridcolor='#252c3a', zerolinecolor='#252c3a')
    fig.update_yaxes(gridcolor='#252c3a', zerolinecolor='#252c3a')
    try:
        return fig.to_json()
    except Exception:
        import json
        return json.dumps({"data": [], "layout": {}})

@app.route("/plot", methods=["POST"])
def plot():
    import plotly.graph_objects as go
    import plotly.subplots as ps
    make_subplots = ps.make_subplots
    data = request.get_json()
    t = data.get("type")
    try:
        if t == "descriptive":
            nums = parse_numbers(data["numbers"])
            fig = make_subplots(rows=1, cols=2, subplot_titles=("Histogram", "Box Plot"))
            fig.add_trace(go.Histogram(x=nums, marker_color='#00d4aa', opacity=0.8, name="Freq"), row=1, col=1)
            fig.add_trace(go.Box(y=nums, marker_color='#ffd166', name="Box", boxmean=True), row=1, col=2)
            fig.update_layout(title="Descriptive Statistics", showlegend=False)
            return jsonify({"plot": fig_json(fig)})

        elif t == "binomial":
            n, p, k = int(data["n"]), float(data["p"]), int(data["k"])
            k_vals = list(range(n + 1))
            def binom_pmf(kk): return math.factorial(n)//(math.factorial(kk)*math.factorial(n-kk)) * (p**kk) * ((1-p)**(n-kk))
            probs = [round(binom_pmf(kk), 6) for kk in k_vals]
            colors = ['#ff6b6b' if kk == k else '#00d4aa' for kk in k_vals]
            fig = go.Figure(go.Bar(x=k_vals, y=probs, marker_color=colors))
            fig.update_layout(title=f"Binomial PMF  n={n}, p={p}", xaxis_title="k", yaxis_title="P(X=k)")
            return jsonify({"plot": fig_json(fig)})

        elif t == "normal":
            mean, std = float(data["mean"]), float(data["std"])
            x1 = float(data["x1"]); tail = data.get("tail","less")
            xs = [mean - 4*std + i*(8*std/400) for i in range(401)]
            def pdf(x): return math.exp(-0.5*((x-mean)/std)**2)/(std*math.sqrt(2*math.pi))
            ys = [pdf(x) for x in xs]
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=xs, y=ys, line=dict(color='#74b9ff', width=2), name="PDF"))
            if tail == "less":
                sx=[x for x in xs if x<=x1]; sy=[pdf(x) for x in sx]
                if sx: fig.add_trace(go.Scatter(x=sx+[sx[-1],sx[0]], y=sy+[0,0], fill='toself', fillcolor='rgba(0,212,170,0.3)', line=dict(width=0), name="Area"))
            elif tail == "greater":
                sx=[x for x in xs if x>=x1]; sy=[pdf(x) for x in sx]
                if sx: fig.add_trace(go.Scatter(x=sx+[sx[-1],sx[0]], y=sy+[0,0], fill='toself', fillcolor='rgba(0,212,170,0.3)', line=dict(width=0), name="Area"))
            elif tail == "between" and data.get("x2"):
                x2=float(data["x2"]); sx=[x for x in xs if x1<=x<=x2]; sy=[pdf(x) for x in sx]
                if sx: fig.add_trace(go.Scatter(x=sx+[sx[-1],sx[0]], y=sy+[0,0], fill='toself', fillcolor='rgba(0,212,170,0.3)', line=dict(width=0), name="Area"))
            fig.update_layout(title=f"Normal Distribution  mu={mean}, sigma={std}", xaxis_title="x", yaxis_title="f(x)")
            return jsonify({"plot": fig_json(fig)})

        elif t in ("regression", "ols"):
            x_vals=parse_numbers(data["x"]); y_vals=parse_numbers(data["y"])
            n=len(x_vals); sx=sum(x_vals); sy=sum(y_vals)
            sxy=sum(a*b for a,b in zip(x_vals,y_vals)); sx2=sum(a**2 for a in x_vals)
            denom=n*sx2-sx**2; b1=(n*sxy-sx*sy)/denom; b0=(sy-b1*sx)/n
            x_line=[min(x_vals),max(x_vals)]; y_line=[b1*x+b0 for x in x_line]
            y_pred=[b1*x+b0 for x in x_vals]; residuals=[y-yp for y,yp in zip(y_vals,y_pred)]
            fig=make_subplots(rows=1,cols=2,subplot_titles=("Scatter + Fit","Residuals"))
            fig.add_trace(go.Scatter(x=x_vals,y=y_vals,mode='markers',marker=dict(color='#00d4aa',size=9),name="Data"),row=1,col=1)
            fig.add_trace(go.Scatter(x=x_line,y=y_line,mode='lines',line=dict(color='#ffd166',width=2),name="Fit"),row=1,col=1)
            fig.add_trace(go.Bar(x=list(range(1,n+1)),y=residuals,marker_color='#ff6b6b',name="Residuals"),row=1,col=2)
            fig.add_hline(y=0,line_dash="dash",line_color="#6b7a99",row=1,col=2)
            fig.update_layout(title="Regression Analysis",showlegend=True)
            return jsonify({"plot": fig_json(fig)})

        elif t == "logistic":
            x_vals=parse_numbers(data["x"]); y_vals=[int(v) for v in parse_numbers(data["y"])]
            n=len(x_vals); b0=0.0; b1=0.0
            def sigmoid(z): return 1/(1+math.exp(-max(-500,min(500,z))))
            for _ in range(1000):
                db0=db1=0
                for x,y in zip(x_vals,y_vals):
                    err=sigmoid(b0+b1*x)-y; db0+=err; db1+=err*x
                b0-=0.1*db0/n; b1-=0.1*db1/n
            xmin=min(x_vals); xmax=max(x_vals)
            xs2=[xmin+i*(xmax-xmin)/200 for i in range(201)]
            ys2=[sigmoid(b0+b1*x) for x in xs2]
            fig=go.Figure()
            fig.add_trace(go.Scatter(x=xs2,y=ys2,line=dict(color='#74b9ff',width=2),name="Sigmoid"))
            colors=['#00d4aa' if y==1 else '#ff6b6b' for y in y_vals]
            fig.add_trace(go.Scatter(x=x_vals,y=[float(y) for y in y_vals],mode='markers',marker=dict(color=colors,size=10),name="Observed"))
            fig.add_hline(y=0.5,line_dash="dash",line_color="#ffd166",annotation_text="Decision Boundary")
            fig.update_layout(title="Logistic Regression",xaxis_title="X",yaxis_title="P(Y=1)")
            return jsonify({"plot": fig_json(fig)})

        elif t == "timeseries":
            nums=parse_numbers(data["numbers"]); ma_order=int(data.get("ma_order",3))
            n=len(nums); t_vals=list(range(1,n+1))
            ma=[sum(nums[i:i+ma_order])/ma_order for i in range(n-ma_order+1)]
            ma_x=list(range(ma_order,n+1))
            st=sum(t_vals); sy2=sum(nums); st2=sum(tt**2 for tt in t_vals); sty=sum(tt*y for tt,y in zip(t_vals,nums))
            denom2=n*st2-st**2; slope=(n*sty-st*sy2)/denom2 if denom2!=0 else 0; intercept=(sy2-slope*st)/n
            trend=[slope*tt+intercept for tt in t_vals]
            fig=go.Figure()
            fig.add_trace(go.Scatter(x=t_vals,y=nums,mode='lines+markers',line=dict(color='#74b9ff'),name="Observed"))
            fig.add_trace(go.Scatter(x=ma_x,y=ma,mode='lines',line=dict(color='#00d4aa',width=2),name=f"MA({ma_order})"))
            fig.add_trace(go.Scatter(x=t_vals,y=trend,mode='lines',line=dict(color='#ffd166',dash='dash'),name="Trend"))
            fig.update_layout(title="Time Series",xaxis_title="Time",yaxis_title="Value")
            return jsonify({"plot": fig_json(fig)})

        elif t == "markov":
            res=calc_markov(data["matrix"],data["steps"],data["init"])
            history=res["history"]; ns=res["n"]
            colors2=['#00d4aa','#74b9ff','#ffd166','#ff6b6b','#bc8cff']
            fig=go.Figure()
            for s in range(ns):
                fig.add_trace(go.Scatter(x=list(range(len(history))),y=[h[s] for h in history],mode='lines+markers',name=f"State {s+1}",line=dict(color=colors2[s%len(colors2)],width=2)))
            fig.update_layout(title="Markov Chain State Probabilities",xaxis_title="Step",yaxis_title="Probability")
            return jsonify({"plot": fig_json(fig)})

        elif t == "survival":
            res=calc_survival(data["times"],data["events"])
            km=res["km_table"]
            times_km=[0]+[float(str(r["time"]).replace('+','')) for r in km]
            s_vals=[1.0]+[r["S_hat_t"] for r in km]
            ct=[float(str(r["time"]).replace('+','')) for r in km if r["deaths_dj"]==0 and r["censored"]>0]
            cs=[r["S_hat_t"] for r in km if r["deaths_dj"]==0 and r["censored"]>0]
            fig=go.Figure()
            fig.add_trace(go.Scatter(x=times_km,y=s_vals,mode='lines',line=dict(color='#00d4aa',width=2,shape='hv'),name="S(t)"))
            if ct: fig.add_trace(go.Scatter(x=ct,y=cs,mode='markers',marker=dict(symbol='cross',size=10,color='#ffd166'),name="Censored"))
            fig.add_hline(y=0.5,line_dash="dash",line_color="#ff6b6b",annotation_text="Median")
            fig.update_layout(title="Kaplan-Meier Survival Curve",xaxis_title="Time",yaxis_title="S(t)",yaxis=dict(range=[0,1.05]))
            return jsonify({"plot": fig_json(fig)})

        elif t == "corrmatrix":
            lines=[l.strip() for l in data.get("datasets","").strip().split('\n') if l.strip()]
            datasets=[parse_numbers(l) for l in lines]
            res=calc_corrmatrix(datasets); mat=res["correlation_matrix"]; k=len(mat)
            labels=[f"V{i+1}" for i in range(k)]
            fig=go.Figure(go.Heatmap(z=mat,x=labels,y=labels,colorscale='RdBu',zmid=0,text=[[str(v) for v in row] for row in mat],texttemplate="%{text}",showscale=True))
            fig.update_layout(title="Correlation Heatmap")
            return jsonify({"plot": fig_json(fig)})

        elif t == "pca":
            lines=[l.strip() for l in data.get("datasets","").strip().split('\n') if l.strip()]
            datasets=[parse_numbers(l) for l in lines]
            res=calc_pca(datasets); labels=[f"V{i+1}" for i in range(res["n_variables"])]; pv=res["proportion_of_variance_%"]
            fig=go.Figure(go.Bar(x=labels,y=pv,marker_color='#bc8cff',text=[f"{v}%" for v in pv],textposition='auto'))
            fig.update_layout(title="PCA Variance Explained (%)",xaxis_title="Variable",yaxis_title="%")
            return jsonify({"plot": fig_json(fig)})

        elif t == "lifetable":
            lx=[int(x) for x in parse_numbers(data["lx"])]; res=calc_life_table(lx)
            ages=[r["age_x"] for r in res["life_table"]]; lx_v=[r["lx"] for r in res["life_table"]]; qx_v=[r["qx"] for r in res["life_table"]]
            fig=make_subplots(rows=1,cols=2,subplot_titles=("Survivors lx","Death Probability qx"))
            fig.add_trace(go.Scatter(x=ages,y=lx_v,fill='tozeroy',line=dict(color='#00d4aa'),name="lx"),row=1,col=1)
            fig.add_trace(go.Bar(x=ages,y=qx_v,marker_color='#ff6b6b',name="qx"),row=1,col=2)
            fig.update_layout(title="Life Table",showlegend=False)
            return jsonify({"plot": fig_json(fig)})

        elif t == "index":
            bp=parse_numbers(data["base_prices"]); cp=parse_numbers(data["current_prices"])
            labels=[f"Item {i+1}" for i in range(len(bp))]
            fig=go.Figure()
            fig.add_trace(go.Bar(x=labels,y=bp,name="Base Price",marker_color='#74b9ff'))
            fig.add_trace(go.Bar(x=labels,y=cp,name="Current Price",marker_color='#ffd166'))
            fig.update_layout(title="Price Comparison",barmode='group',xaxis_title="Item",yaxis_title="Price")
            return jsonify({"plot": fig_json(fig)})

        elif t == "ci":
            nums=parse_numbers(data["numbers"]); conf=int(data["confidence"])
            res=calc_ci(nums,conf); mean=res["sample_mean"]; lo=res["lower_bound"]; hi=res["upper_bound"]
            fig=go.Figure()
            fig.add_trace(go.Scatter(x=nums,y=[0]*len(nums),mode='markers',marker=dict(color='#74b9ff',size=8,opacity=0.6),name="Data"))
            fig.add_shape(type="line",x0=lo,x1=hi,y0=0,y1=0,line=dict(color='#00d4aa',width=6))
            fig.add_vline(x=mean,line_color='#ffd166',line_dash='dash',annotation_text="Mean")
            fig.add_vline(x=lo,line_color='#00d4aa',annotation_text="Lower")
            fig.add_vline(x=hi,line_color='#00d4aa',annotation_text="Upper")
            fig.update_layout(title=f"{conf}% Confidence Interval",xaxis_title="Value",yaxis=dict(visible=False))
            return jsonify({"plot": fig_json(fig)})

        elif t in ("ztest","ttest"):
            if t=="ztest":
                sm=float(data["sample_mean"]); pm=float(data["pop_mean"]); ps=float(data["pop_std"]); n=int(data["n"])
                se=ps/math.sqrt(n); stat=(sm-pm)/se; stat_name="Z"
            else:
                nums=parse_numbers(data["numbers"]); pm=float(data["pop_mean"])
                xbar=statistics.mean(nums); s=statistics.stdev(nums); se=s/math.sqrt(len(nums)); stat=(xbar-pm)/se; stat_name="t"
            xs=[-4+i*8/400 for i in range(401)]
            ys=[math.exp(-0.5*x**2)/math.sqrt(2*math.pi) for x in xs]
            fig=go.Figure()
            fig.add_trace(go.Scatter(x=xs,y=ys,line=dict(color='#74b9ff',width=2),name="Distribution"))
            cr=1.96
            rx1=[x for x in xs if x<=-cr]; ry1=[math.exp(-0.5*x**2)/math.sqrt(2*math.pi) for x in rx1]
            rx2=[x for x in xs if x>=cr];  ry2=[math.exp(-0.5*x**2)/math.sqrt(2*math.pi) for x in rx2]
            if rx1: fig.add_trace(go.Scatter(x=rx1+[rx1[-1],rx1[0]],y=ry1+[0,0],fill='toself',fillcolor='rgba(255,107,107,0.3)',line=dict(width=0),name="Reject"))
            if rx2: fig.add_trace(go.Scatter(x=rx2+[rx2[-1],rx2[0]],y=ry2+[0,0],fill='toself',fillcolor='rgba(255,107,107,0.3)',line=dict(width=0),showlegend=False))
            fig.add_vline(x=stat,line_color='#ffd166',line_dash='dash',annotation_text=f"{stat_name}={round(stat,3)}")
            fig.update_layout(title=f"{stat_name}-Test",xaxis_title=stat_name,yaxis_title="Density")
            return jsonify({"plot": fig_json(fig)})

        return jsonify({"error": f"No plot for type: {t}"})
    except Exception as e:
        return jsonify({"error": str(e)})
