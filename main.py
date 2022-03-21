import os
import utils
import numpy as np
import pandas as pd
import scipy.optimize as sco
from functools import partial
from itertools import groupby


def main():
    data = utils.load_data('Python assessment', 'Start Universe')
    constituents = question1(data)
    portfolios, objectives, sector_data = question2(data, constituents)

    # print results to terminal
    for i, portfolio in enumerate(portfolios):
        print(f"Portfolio {i + 1}: \n")
        print(portfolio)
        print("\n")
        print(f"Value of objective function for rebalance {i + 1}: {objectives[i]} \n")
        print("\n")
        print("Sector Breakdown: \n")
        print(sector_data[i])
        print("\n")

    # export data to excel file
    with pd.ExcelWriter(os.path.join('output', "Output.xlsx")) as writer:
        for i, portfolio in enumerate(portfolios):
            ref_date = str(portfolio["Ref Date"][0].date())
            portfolio.to_excel(writer, sheet_name=f"{ref_date} - Constituents", index=False)
            sector_data[i].to_excel(writer, sheet_name=f"{ref_date} - Sectors", index=False)


def question1(data):
    """This function derives the constituent assets for each portfolio subject to the stipulated construction rules"""
    # split data into sub-frames by date
    data = data.sort_values('Ref Date', axis=0, ascending=True, ignore_index=True)
    dates = data['Ref Date']
    unique_dates = list()
    for date, items in groupby(dates):
        unique_dates.append(date)

    sub_frames = list()
    for date in unique_dates:
        sub_frame = data.loc[data['Ref Date'] == date]
        sub_frames.append(sub_frame)

    A, B, C = sub_frames

    # construct portfolios subject to construction rules
    portfolios = []
    for x, y in enumerate([A, B, C]):
        y = y.sort_values('Z_Value', axis=0, ascending=False)
        portfolio = y.head(40)
        y = y.tail(len(y) - len(portfolio))
        y.reset_index(inplace=True)
        if x > 0:
            R = y.head(20)
            S = portfolios[x - 1]
            inter = R.loc[R["RIC"].isin(list(S["RIC"]))]
            inter = inter[portfolio.columns]
            portfolio = portfolio.append(inter.head(10), ignore_index=True)
            y = y.drop(list(inter.index.values), axis=0)
            y.reset_index(inplace=True)

        if len(portfolio) < 50:
            portfolio = portfolio.append(y.loc[0:50 - len(portfolio) - 1][portfolio.columns], ignore_index=True)
        portfolios.append(portfolio)

    return portfolios


def question2(data, constituents):
    """This function derives the constrained and unconstrained weights for each portfolio"""
    portfolios = []
    objectives = []
    sector_df = []

    sectors = data["Sector Code"]
    sector_codes = []
    for code, rows in groupby(sectors):
        if code not in sector_codes:
            sector_codes.append(code)

    sector_codes.sort()

    for portfolio in constituents:
        # Compute unconstrained weights
        FCap = np.array(portfolio["FCap Wt"])
        Z = np.array(portfolio["Z_Value"])
        Wu = pd.DataFrame(np.multiply(FCap, 1 + Z), columns=["Unconstrained Weights"])
        Wu = Wu * (1 / np.sum(Wu))
        portfolio["Unconstrained Weights"] = Wu

        # Define upper and lower parameter bounds
        upper_bounds = np.minimum(0.05 * np.ones(len(FCap)), 20 * FCap)
        bounds = tuple()
        for i in range(len(upper_bounds)):
            bounds = bounds + ((0.0005, upper_bounds[i]),)

        # Define sector constraints
        selection_vectors = []
        for code in sector_codes:
            rows = np.array(portfolio["Sector Code"] == code).astype(int)
            selection_vectors.append(rows)

        cons = list()
        for x, y in enumerate(selection_vectors):
            cons.append({'type': 'ineq', 'fun': partial(utils.f_constraint, y)})

        # Ensure weights sum to 1
        cons.append({'type': 'eq', 'fun': lambda w: 1 - np.inner(w, np.ones(len(w)))})

        # Set initial weight vector equal to unrestricted weight vector
        init = np.array(Wu).reshape(len(Wu))

        # Minimise objective function subject to bounds and constraints using Sequential Least Squares
        result = sco.minimize(utils.target, x0=init, args=init, method='SLSQP', bounds=bounds, constraints=cons)

        portfolio["Constrained Weights"] = result['x']
        portfolio = portfolio.sort_values('Z_Value', axis=0, ascending=False, ignore_index=True)
        portfolios.append(portfolio)

        objectives.append(result['fun'])

        # Construct Sector Data
        sector_data = pd.DataFrame(sector_codes, columns=["Sector Code"])
        sector_data["Num Stocks"] = [np.sum(selection_vectors[i]) for i in range(len(selection_vectors))]
        sector_data["Uncapped Weights"] = [np.sum(portfolio.loc[portfolio["Sector Code"] == code]["Unconstrained "
                                                                                                  "Weights"]) for
                                           code in sector_codes]
        sector_data["Capped Weights"] = [np.sum(portfolio.loc[portfolio["Sector Code"] == code]["Constrained "
                                                                                                "Weights"]) for
                                         code in sector_codes]
        sector_data["Max Weight"] = [np.minimum(np.inner(vector, upper_bounds), 0.5) for vector in selection_vectors]

        sector_df.append(sector_data)

    return portfolios, objectives, sector_df


if __name__ == '__main__':
    main()
