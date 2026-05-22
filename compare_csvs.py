#!/usr/bin/env python3
"""Confronta due file CSV (test_results_davide.csv e results.csv).

Output:
- stampa riepilogo del confronto
- scrive `davide_better_cases.csv` con i casi in cui `Solution` < `ga_cost`
- scrive un log `compare_log.txt` con i dettagli sui tempi di esecuzione

Uso:
    python compare_csvs.py
    oppure
    python compare_csvs.py --mine results.csv --theirs test_results_davide.csv
"""
import csv
import argparse
from pathlib import Path


def read_csv_as_dict(path, key_cols):
    rows = {}
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for r in reader:
            key = tuple(r[c].strip() for c in key_cols)
            rows[key] = {k: v.strip() for k, v in r.items()}
    return rows


def to_float(s):
    try:
        return float(s.replace(',', ''))
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mine', default='results.csv', help='Il file CSV dei miei risultati (default: results.csv)')
    parser.add_argument('--theirs', default='test_results_davide.csv', help='Il file CSV di Davide (default: test_results_davide.csv)')
    parser.add_argument('--out', default='davide_better_cases.csv', help='CSV output con i casi in cui Davide è migliore')
    parser.add_argument('--log', default='compare_log.txt', help='File di log con i dettagli sui tempi')
    args = parser.parse_args()

    mine_path = Path(args.mine)
    theirs_path = Path(args.theirs)

    if not mine_path.exists():
        print(f"File non trovato: {mine_path}")
        return
    if not theirs_path.exists():
        print(f"File non trovato: {theirs_path}")
        return

    # Chiavi per abbinare le configurazioni
    key_cols_mine = ['size', 'density', 'alpha', 'beta']
    # Davide usa intestazioni con maiuscole: Size,Density,Alpha,Beta
    key_cols_theirs = ['Size', 'Density', 'Alpha', 'Beta']

    mine = read_csv_as_dict(mine_path, key_cols_mine)
    theirs = read_csv_as_dict(theirs_path, key_cols_theirs)

    total = 0
    davide_better = []
    davide_better_count = 0
    mine_better_count = 0
    equal_count = 0
    missing_in_mine = 0
    missing_in_theirs = 0

    log_lines = []

    for key, th_row in theirs.items():
        total += 1
        # try to find matching key in mine: mine keys are lowercase strings
        mine_key = (key[0], key[1], key[2], key[3])

        if mine_key not in mine:
            missing_in_mine += 1
            log_lines.append(f"MANCANTE_MINE: {mine_key}")
            continue

        my_row = mine[mine_key]

        # estrai costi
        # mine: 'ga_cost' ; theirs: 'Solution' or 'Solution'
        my_cost = to_float(my_row.get('ga_cost') or my_row.get('ga cost') or my_row.get('ga_cost', ''))
        their_cost = to_float(th_row.get('Solution') or th_row.get('solution') or th_row.get('Solution', ''))

        # tempi: mine has 'elapsed_time_min' (minutes), theirs has 'Elapsed' (seconds?)
        my_time_min = to_float(my_row.get('elapsed_time_min') or my_row.get('elapsed_time') or '')
        their_time = to_float(th_row.get('Elapsed') or th_row.get('elapsed') or '')

        # convert mine time to seconds if provided in minutes
        my_time_sec = None
        if my_time_min is not None:
            my_time_sec = my_time_min * 60

        if my_cost is None or their_cost is None:
            log_lines.append(f"SKIP_COST_MISSING: {mine_key} my_cost={my_cost} their_cost={their_cost}")
            continue

        if their_cost < my_cost:
            davide_better_count += 1
            davide_better.append((mine_key, my_cost, their_cost, my_time_sec, their_time, my_row.get('is_valid'), th_row.get('Status')))
            log_lines.append(f"DAVIDE_BETTER: {mine_key} davide={their_cost} mine={my_cost}")
        elif their_cost > my_cost:
            mine_better_count += 1
            log_lines.append(f"MINE_BETTER: {mine_key} davide={their_cost} mine={my_cost}")
        else:
            equal_count += 1
            log_lines.append(f"EQUAL: {mine_key} both={my_cost}")

        # tempi confronto
        if my_time_sec is not None and their_time is not None:
            if their_time < my_time_sec:
                log_lines.append(f"TIME: {mine_key} davide faster by {my_time_sec - their_time:.3f}s ({their_time:.3f}s vs {my_time_sec:.3f}s)")
            else:
                log_lines.append(f"TIME: {mine_key} mine faster by {their_time - my_time_sec:.3f}s ({their_time:.3f}s vs {my_time_sec:.3f}s)")

    # Check keys present in mine but missing in theirs
    for key in mine.keys():
        if key not in theirs:
            missing_in_theirs += 1

    # Scrivi CSV dei casi vincenti di Davide
    out_path = Path(args.out)
    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['size', 'density', 'alpha', 'beta', 'my_ga_cost', 'davide_solution', 'my_time_sec', 'davide_time', 'my_status', 'davide_status'])
        for item in davide_better:
            key, myc, thc, mts, tht, myst, thst = item
            writer.writerow([key[0], key[1], key[2], key[3], myc, thc, mts if mts is not None else '', tht if tht is not None else '', myst or '', thst or ''])

    # Scrivi log
    with open(args.log, 'w', encoding='utf-8') as f:
        f.write(f"Total compared (theirs rows): {total}\n")
        f.write(f"Davide better count: {davide_better_count}\n")
        f.write(f"Mine better count: {mine_better_count}\n")
        f.write(f"Equal count: {equal_count}\n")
        f.write(f"Missing in mine: {missing_in_mine}\n")
        f.write(f"Missing in theirs: {missing_in_theirs}\n")
        f.write('\nDetailed lines:\n')
        for L in log_lines:
            f.write(L + '\n')

    # Stampe a schermo
    print(f"Totale righe di Davide lette: {total}")
    print(f"Davide migliore in {davide_better_count} casi -> scritto in {out_path}")
    print(f"Mio migliore in {mine_better_count} casi")
    print(f"Uguali: {equal_count}")
    print(f"Mancanti in mio file: {missing_in_mine}")
    print(f"Mancanti in file di Davide: {missing_in_theirs}")
    print(f"Log dettagliato: {args.log}")


if __name__ == '__main__':
    main()
