import sympy as sp

def get_s(t:int):
    s = f"H(w{t-1}) * (S{t-1} + 1)"
    return (sp.parse_expr(f"S{t}"), sp.parse_expr(s))

def formula(T:int):
    s_last, e_last = get_s(T)

    for t in range(T, -1, -1):
        m = {}
        if t == 0:
            s = sp.Symbol(f"S{t}")
            e = sp.parse_expr("0")
        else:
            s, e = get_s(t)
        m[s] = e
        e_last = e_last.xreplace(m)
        print(m)

    print("Orig: ", s_last, "=", e_last)
    print("Expn: ", s_last, "=", sp.expand(e_last))

    s = sp.expand(e_last)
    s = str(s)
    s = s.replace("H", "ste_binarize01")
    s = s.replace("w", "self.weight_shifts[")
    print("Code: ", s_last, "=", s)



if __name__ == '__main__':
    formula(8)