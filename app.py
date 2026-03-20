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
        "formula": f"C({n},{k}) × {p}^{k} × {1-p}^{n-k}"
    }

# ── 4. Normal Distribution (AST 102, AST 201) ────────────────────────────────
def calc_normal(mean, std, x1, x2=None, tail="less"):
    mean, std = float(mean), float(std)
    x1 = float(x1)
    def phi(x): return 0.5 * (1 + math.erf((x - mean) / (std * math.sqrt(2))))
    if tail == "less":
        prob = phi(x1); z = (x1 - mean) / std
        return {"P(X < x1)": safe(prob), "z_score": safe(z)}
    elif tail == "greater":
        prob = 1 - phi(x1); z = (x1 - mean) / std
        return {"P(X > x1)": safe(prob), "z_score": safe(z)}
    elif tail == "between" and x2 is not None:
        x2 = float(x2)
        prob = phi(x2) - phi(x1)
        return {"P(x1<X<x2)": safe(prob), "z1": safe((x1-mean)/std), "z2": safe((x2-mean)/std)}
    return {"error": "Invalid parameters."}

# ── 5. Regression (AST 303) ──────────────────────────────────────────────────
def calc_regression(x_vals, y_vals):
    n = len(x_vals)
    if n != len(y_vals) or n < 2: return {"error": "X and Y must have equal count (≥ 2)."}
    sx = sum(x_vals); sy = sum(y_vals)
    sxy = sum(x*y for x,y in zip(x_vals, y_vals))
    sx2 = sum(x**2 for x in x_vals); sy2 = sum(y**2 for y in y_vals)
    denom = n*sx2 - sx**2
    if denom == 0: return {"error": "All X values identical — regression undefined."}
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
        "equation": f"ŷ = {round(b1,4)}x + {round(b0,4)}",
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
        "reject_H0_at_5%": "Yes ✓" if p < 0.05 else "No ✗",
        "reject_H0_at_1%": "Yes ✓" if p < 0.01 else "No ✗",
    }

# ── 7. One-Sample t-Test (AST 203) ───────────────────────────────────────────
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
    xbar = statistics.mean(nums)
    s = statistics.stdev(nums)
    se = s / math.sqrt(n)
    t = (xbar - pop_mean) / se
    df = n - 1
    x = df/(df + t*t)
    p = min(_betainc(df/2, 0.5, x), 1.0)
    return {
        "sample_mean": safe(xbar), "sample_std": safe(s),
        "standard_error": safe(se), "t_statistic": safe(t),
        "degrees_of_freedom": df, "p_value_approx_two_tailed": safe(p),
        "reject_H0_at_5%": "Yes ✓" if p < 0.05 else "No ✗",
    }

# ── 8. Confidence Interval (AST 203) ─────────────────────────────────────────
def calc_ci(nums, conf):
    n = len(nums)
    if n < 2: return {"error": "Need at least 2 observations."}
    mean = statistics.mean(nums)
    se = statistics.stdev(nums) / math.sqrt(n)
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
    except Exception as e:
        result = {"error": str(e)}
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True)
