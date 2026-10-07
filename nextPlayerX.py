import json
import numpy as np
import pandas as pd
import seaborn as sns
from datetime import datetime
import matplotlib.pyplot as plt
from safe_requests import safeRequest
from sklearn.decomposition import PCA
from matplotlib.patches import Polygon

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 2000)


def jsonRequest(url, cache_ttl=None, permanent=None):
    if cache_ttl is None:
        resp = safeRequest(url, permanent=permanent)  # uses default CACHE_TTL
    else:
        cache_ttl = cache_ttl * 3600
        resp = safeRequest(url, ttl=cache_ttl, permanent=permanent)
    return json.loads(resp.text)


def calcPer90(targetSeries, minutesPlayedSeries):
    per90 = targetSeries / minutesPlayedSeries * 90
    return per90


def plotPolynomialRegression(df, metricColumn, polynomialDegree):
    # 1. Clean the dataframe to handle missing or infinite values
    cleanDf = df.copy()
    xData = cleanDf['teamAvgPossession']
    yData = cleanDf[metricColumn]

    # 2. Set up the figure
    plt.figure(figsize=(10, 6))

    # 3. Create the scatter plot of the raw data points
    plt.scatter(xData, yData, color='#33b5e5', alpha=0.6, edgecolors='black', label='Players')

    # 4. Calculate the polynomial coefficients
    coefficients = np.polyfit(xData, yData, polynomialDegree)
    polynomialEquation = np.poly1d(coefficients)

    # 5. Generate a smooth line for the curve
    # We use linspace to create 100 evenly spaced points between the min and max possession
    xLine = np.linspace(xData.min(), xData.max(), 100)
    yLine = polynomialEquation(xLine)

    # 6. Plot the polynomial curve
    plt.plot(xLine, yLine, color='red', linewidth=2.5, label=f'Polynomial Fit (Degree {polynomialDegree})')

    # 7. Format the chart
    plt.title(f'Team Possession vs {metricColumn}')
    plt.xlabel('Team Average Possession')
    plt.ylabel(metricColumn)
    plt.legend(loc='best')
    plt.grid(axis='both', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.show()

    yPredicted = polynomialEquation(xData)

    # 2. Calculate the residual sum of squares
    ssResidual = np.sum((yData - yPredicted) ** 2)

    # 3. Calculate the total sum of squares
    ssTotal = np.sum((yData - np.mean(yData)) ** 2)

    # 4. Calculate R^2
    rSquared = 1 - (ssResidual / ssTotal)

    print(f"{metricColumn} R^2 Score (Degree {polynomialDegree}): {rSquared:.4f}")


leagueIds = [17, 8, 23, 35, 34]
seasonIds = [(76986, 61627, 52186, 41886), (77559, 61643, 52376, 42409), (76457, 63515, 52760, 42415),
             (77333, 63516, 52608, 42268), (77356, 61736, 52571, 42273)]
seasonKey = {
    0: "25/26",
    1: "24/25",
    2: "23/24",
    3: "22/23",
}

positionKey = {
    "ST": "F",
    "LW": "F~M",
    "RW": "F~M",
    "AM": "M",
    "ML": "M",
    "MC": "M",
    "MR": "M",
    "DM": "M",
    "DL": "M~D",
    "DC": "D",
    "DR": "M~D"
}
currentDate = datetime.now()
residualColumnsDC = ['totalPassesPer90', 'accuratePassesPercentage', 'totalOwnHalfPassesPer90', 'ownHalfPassSuccess',
                     'totalOppositionHalfPassesPer90', 'oppHalfPassSuccess', 'accurateFinalThirdPassesPer90',
                     'accurateLongBallsPercentage', 'clearancesPer90', 'chippedPassSuccess', 'touchesPer90']
residualColumnsDFB = ['expectedAssistsPer90', 'totalPassesPer90', 'accuratePassesPercentage', 'totalOwnHalfPassesPer90',
                      'ownHalfPassSuccess', 'totalOppositionHalfPassesPer90', 'oppHalfPassSuccess',
                      'accurateFinalThirdPassesPer90', 'accurateLongBallsPercentage', 'touchesPer90']
residualColumnsDM = ['expectedAssistsPer90', 'totalPassesPer90', 'accuratePassesPercentage', 'totalOwnHalfPassesPer90',
                     'ownHalfPassSuccess', 'totalOppositionHalfPassesPer90', 'oppHalfPassSuccess',
                     'accurateFinalThirdPassesPer90', 'accurateLongBallsPercentage', 'touchesPer90']
residualColumnsMC = ['expectedAssistsPer90', 'totalPassesPer90', 'accuratePassesPercentage', 'totalOwnHalfPassesPer90',
                     'ownHalfPassSuccess', 'totalOppositionHalfPassesPer90', 'oppHalfPassSuccess',
                     'accurateFinalThirdPassesPer90', 'accurateLongBallsPercentage', 'touchesPer90']
residualColumnsMW = ['expectedAssistsPer90', 'totalPassesPer90', 'accuratePassesPercentage', 'totalOwnHalfPassesPer90',
                     'ownHalfPassSuccess', 'totalOppositionHalfPassesPer90', 'oppHalfPassSuccess',
                     'accurateFinalThirdPassesPer90', 'touchesPer90']
residualColumnsAM = ['expectedGoalsPer90', 'bigChancesCreatedPer90', 'assistsPer90', 'expectedAssistsPer90',
                     'totalPassesPer90', 'accuratePassesPercentage', 'ownHalfPassSuccess',
                     'totalOppositionHalfPassesPer90', 'oppHalfPassSuccess', 'accurateFinalThirdPassesPer90',
                     'shotsFromInsideTheBoxPer90', 'touchesPer90']
residualColumnsW = ['goalsPer90', 'expectedGoalsPer90', 'bigChancesCreatedPer90', 'assistsPer90',
                    'expectedAssistsPer90', 'totalPassesPer90', 'accuratePassesPercentage', 'ownHalfPassSuccess',
                    'totalOppositionHalfPassesPer90', 'oppHalfPassSuccess', 'accurateFinalThirdPassesPer90',
                    'keyPassesPer90', 'totalShotsPer90', 'totalAerialDuelsPer90', 'goalsFromInsideTheBoxPer90',
                    'shotsFromInsideTheBoxPer90', 'touchesPer90', 'totalAttemptAssistPer90']
residualColumnsST = ['goalsPer90', 'expectedGoalsPer90', 'expectedAssistsPer90', 'accuratePassesPercentage',
                     'oppHalfPassSuccess', 'accurateFinalThirdPassesPer90', 'totalShotsPer90', 'totalAerialDuelsPer90',
                     'totalDuelsPer90', 'goalsFromInsideTheBoxPer90', 'shotsFromInsideTheBoxPer90']

residualColumnsKey = {
    "DC": residualColumnsDC,
    "DR": residualColumnsDFB,
    "DL": residualColumnsDFB,
    "DM": residualColumnsDM,
    "MC": residualColumnsMC,
    "MR": residualColumnsMW,
    "ML": residualColumnsMW,
    "AM": residualColumnsAM,
    "RW": residualColumnsW,
    "LW": residualColumnsW,
    "ST": residualColumnsST
}


def findSimilarPlayers(position: str, targetPlayer: str, columnsToResidualise: list, yearString: str, maxAge: int,
                       maxMarketValue: int, displayVersion: str, specificSeason: str = "",
                       checkCorrelations: bool = False, comparisonGraph: bool = False, getData: bool = False):
    if getData:
        third = positionKey[position]
        playerSeasonStats = {}
        for x in range(len(leagueIds)):
            leagueId = leagueIds[x]
            for y in range(len(seasonIds[x])):
                seasonId = seasonIds[x][y]
                seasonName = seasonKey[y]
                data = jsonRequest(
                    f"/api/v1/unique-tournament/{leagueId}/season/{seasonId}/statistics?limit=100&order=-minutesPlayed&accumulation=total&fields=minutesPlayed%2Cappearances&filters=appearances.GT.12%2Cposition.in.{third}",
                    cache_ttl=0)
                seasonPlayers = data['results']
                if data['page'] != data['pages']:
                    data = jsonRequest(
                        f"/api/v1/unique-tournament/{leagueId}/season/{seasonId}/statistics?limit=100&order=-minutesPlayed&offset=100&accumulation=total&fields=minutesPlayed%2Cappearances&filters=appearances.GT.12%2Cposition.in.{third}",
                        cache_ttl=0)
                    seasonPlayers.extend(data['results'])
                if data['page'] != data['pages']:
                    data = jsonRequest(
                        f"/api/v1/unique-tournament/{leagueId}/season/{seasonId}/statistics?limit=100&order=-minutesPlayed&offset=200&accumulation=total&fields=minutesPlayed%2Cappearances&filters=appearances.GT.12%2Cposition.in.{third}",
                        cache_ttl=0)
                    seasonPlayers.extend(data['results'])
                for player in seasonPlayers:
                    playerId = player['player']['id']
                    playerName = player['player']['name']
                    playerCurrentSummary = jsonRequest(f"/api/v1/player/{playerId}", cache_ttl=0)['player']
                    if 'retired' in playerCurrentSummary:
                        if playerCurrentSummary['retired']:
                            continue
                    playerDOB = playerCurrentSummary['dateOfBirthTimestamp']
                    birthDate = datetime.fromtimestamp(playerDOB)
                    currentAge = currentDate.year - birthDate.year - (
                            (currentDate.month, currentDate.day) < (birthDate.month, birthDate.day))
                    if 'proposedMarketValue' in playerCurrentSummary:
                        playerMarketValueEst = playerCurrentSummary['proposedMarketValue']
                    else:
                        continue
                    if 'positionsDetailed' in playerCurrentSummary:
                        if position not in playerCurrentSummary['positionsDetailed']:
                            continue
                    else:
                        continue
                    seasonTotal = jsonRequest(
                        f"/api/v1/player/{playerId}/unique-tournament/{leagueId}/season/{seasonId}/statistics/overall",
                        cache_ttl=0)
                    seasonStatsTotal = seasonTotal['statistics']
                    teamId = seasonTotal['team']['id']
                    teamName = seasonTotal['team']['shortName']
                    categories = ['goals', 'bigChancesCreated', 'bigChancesMissed', 'assists', 'expectedAssists',
                                  'goalsAssistsSum', 'accuratePasses', 'inaccuratePasses', 'totalPasses',
                                  'accuratePassesPercentage', 'accurateOwnHalfPasses', 'accurateOppositionHalfPasses',
                                  'accurateFinalThirdPasses', 'keyPasses', 'successfulDribbles',
                                  'successfulDribblesPercentage', 'tackles', 'interceptions', 'yellowCards',
                                  'directRedCards', 'redCards', 'accurateCrosses', 'accurateCrossesPercentage',
                                  'totalShots', 'shotsOnTarget', 'shotsOffTarget', 'groundDuelsWon',
                                  'groundDuelsWonPercentage', 'aerialDuelsWon', 'aerialDuelsWonPercentage',
                                  'totalDuelsWon',
                                  'totalDuelsWonPercentage', 'minutesPlayed', 'goalConversionPercentage',
                                  'penaltiesTaken',
                                  'penaltyGoals', 'penaltyWon', 'penaltyConceded', 'shotFromSetPiece', 'freeKickGoal',
                                  'goalsFromInsideTheBox', 'goalsFromOutsideTheBox', 'shotsFromInsideTheBox',
                                  'shotsFromOutsideTheBox', 'headedGoals', 'accurateLongBalls',
                                  'accurateLongBallsPercentage', 'clearances', 'errorLeadToGoal', 'errorLeadToShot',
                                  'dispossessed', 'possessionLost', 'possessionWonAttThird', 'totalChippedPasses',
                                  'accurateChippedPasses', 'touches', 'wasFouled', 'fouls', 'hitWoodwork', 'ownGoals',
                                  'dribbledPast', 'offsides', 'blockedShots', 'passToAssist', 'matchesStarted',
                                  'setPieceConversion', 'totalAttemptAssist', 'totalContest', 'totalCross', 'duelLost',
                                  'aerialLost', 'totalLongBalls', 'tacklesWon', 'tacklesWonPercentage',
                                  'yellowRedCards',
                                  'totalOwnHalfPasses', 'totalOppositionHalfPasses', 'expectedGoals', 'ballRecovery',
                                  'outfielderBlocks', 'appearances']
                    for key in seasonStatsTotal.copy().keys():
                        if key not in categories:
                            seasonStatsTotal.pop(key)
                    seasonStatsTotal['id'] = playerId
                    seasonStatsTotal['teamId'] = teamId
                    seasonStatsTotal['seasonId'] = seasonId
                    seasonStatsTotal['leagueId'] = leagueId
                    seasonStatsTotal['currentAge'] = currentAge
                    seasonStatsTotal['marketValue'] = playerMarketValueEst
                    index = f"{playerName} @ {teamName} {seasonName}"
                    playerSeasonStats[index] = seasonStatsTotal
                    print(index)

        with open(f"data/{position}Seasons.json", "w", encoding="utf-8") as f:
            json.dump(playerSeasonStats, f, indent=4)

        teamSeasonStats = {}
        for x in range(len(leagueIds)):
            leagueId = leagueIds[x]
            for y in range(len(seasonIds[x])):
                seasonId = seasonIds[x][y]
                seasonName = seasonKey[y]
                data = \
                    jsonRequest(f"/api/v1/unique-tournament/{leagueId}/season/{seasonId}/top-teams/overall",
                                cache_ttl=0)[
                        'topTeams'][
                        'averageBallPossession']
                for z in data:
                    avgPossession = z['statistics']['averageBallPossession']
                    teamName = z['team']['name']
                    teamId = z['team']['id']
                    index = f"{teamName} {seasonName}"
                    teamSeasonStats[index] = {
                        "teamId": teamId,
                        "seasonId": seasonId,
                        "leagueId": leagueId,
                        "avgBallPossession": avgPossession
                    }

        with open("data/teamSeasonPossession.json", "w", encoding="utf-8") as f:
            json.dump(teamSeasonStats, f, indent=4)

    with open(f"data/{position}Seasons.json", "r", encoding="utf-8") as f:
        playerData = json.load(f)

    with open("data/teamSeasonPossession.json", "r", encoding="utf-8") as f:
        teamSeasonData = json.load(f)

    playerDF = pd.DataFrame(playerData).T

    possessionDF = pd.DataFrame(teamSeasonData).T

    mergedDf = pd.merge(
        playerDF.reset_index(),
        possessionDF.reset_index(),
        on=['teamId', 'seasonId', 'leagueId'],
        how='left'
    )

    mergedDf.set_index('index_x', inplace=True)
    mergedDf.index.name = None

    mergedDf.rename(columns={'avgBallPossession': 'teamAvgPossession'}, inplace=True)

    finalColumns = ['goalsPer90', 'expectedGoalsPer90', 'bigChancesCreatedPer90', 'assistsPer90',
                    'expectedAssistsPer90',
                    'totalPassesPer90', 'accuratePassesPercentage', 'totalOwnHalfPassesPer90', 'ownHalfPassSuccess',
                    'totalOppositionHalfPassesPer90', 'oppHalfPassSuccess', 'accurateFinalThirdPassesPer90',
                    'keyPassesPer90', 'totalDribblesPer90', 'successfulDribblesPercentage', 'tacklesPer90',
                    'tacklesWonPercentage', 'interceptionsPer90', 'yellowCardsPer90', 'redCardsPer90',
                    'totalCrossPer90',
                    'accurateCrossesPercentage', 'totalShotsPer90', 'shotOnTargetPercentage', 'totalGroundDuelsPer90',
                    'groundDuelsWonPercentage', 'totalAerialDuelsPer90', 'aerialDuelsWonPercentage', 'totalDuelsPer90',
                    'totalDuelsWonPercentage', 'shotFromSetPiecePer90', 'goalsFromInsideTheBoxPer90',
                    'goalsFromOutsideTheBoxPer90', 'shotsFromInsideTheBoxPer90', 'shotsFromOutsideTheBoxPer90',
                    'headedGoalsPer90', 'totalLongBallsPer90', 'accurateLongBallsPercentage', 'clearancesPer90',
                    'errorLeadToGoalPer90', 'errorLeadToShotPer90', 'dispossessedPer90', 'possessionLostPer90',
                    'possessionWonAttThirdPer90', 'totalChippedPassesPer90', 'chippedPassSuccess', 'touchesPer90',
                    'wasFouledPer90', 'foulsPer90', 'dribbledPastPer90', 'totalAttemptAssistPer90', 'minutesPerApp',
                    'currentAge', 'marketValue', 'teamAvgPossession']

    mergedDf['ownHalfPassSuccess'] = (mergedDf['accurateOwnHalfPasses'] / mergedDf['totalOwnHalfPasses']) * 100
    mergedDf['oppHalfPassSuccess'] = (mergedDf['accurateOppositionHalfPasses'] / mergedDf[
        'totalOppositionHalfPasses']) * 100
    mergedDf['totalDribbles'] = mergedDf['successfulDribbles'] / (mergedDf['successfulDribblesPercentage'] / 100)
    mergedDf['totalGroundDuels'] = mergedDf['groundDuelsWon'] / (mergedDf['groundDuelsWonPercentage'] / 100)
    mergedDf['totalAerialDuels'] = mergedDf['aerialDuelsWon'] + mergedDf['aerialLost']
    mergedDf['totalDuels'] = mergedDf['totalDuelsWon'] + mergedDf['duelLost']
    mergedDf['chippedPassSuccess'] = (mergedDf['accurateChippedPasses'] / mergedDf['totalChippedPasses']) * 100
    mergedDf['shotOnTargetPercentage'] = (mergedDf['shotsOnTarget'] / mergedDf['totalShots']) * 100
    mergedDf['minutesPerApp'] = mergedDf['minutesPlayed'] / mergedDf['appearances']

    columnsToPer90 = ['goals', 'expectedGoals', 'bigChancesCreated', 'assists', 'expectedAssists', 'totalPasses',
                      'totalOwnHalfPasses', 'totalOppositionHalfPasses', 'accurateFinalThirdPasses', 'keyPasses',
                      'totalDribbles', 'tackles', 'interceptions', 'yellowCards', 'redCards', 'totalCross',
                      'totalShots',
                      'totalGroundDuels', 'totalAerialDuels', 'totalDuels', 'shotFromSetPiece', 'goalsFromInsideTheBox',
                      'goalsFromOutsideTheBox', 'shotsFromInsideTheBox', 'shotsFromOutsideTheBox', 'headedGoals',
                      'totalLongBalls', 'clearances', 'errorLeadToGoal', 'errorLeadToShot', 'dispossessed',
                      'possessionLost', 'possessionWonAttThird', 'totalChippedPasses', 'touches', 'wasFouled', 'fouls',
                      'dribbledPast', 'totalAttemptAssist']

    for col in columnsToPer90:
        mergedDf[f'{col}Per90'] = calcPer90(mergedDf[col], mergedDf['minutesPlayed'])

    mergedDf = mergedDf[finalColumns].fillna(0)

    if checkCorrelations:
        for column in finalColumns:
            plotPolynomialRegression(mergedDf, column, 3)
        return None

    residualDf = mergedDf.copy()

    for column in columnsToResidualise:
        xData = residualDf['teamAvgPossession']
        yData = residualDf[column]

        # Calculate the polynomial coefficients and generate the equation
        coefficients = np.polyfit(xData, yData, 3)
        polynomialEquation = np.poly1d(coefficients)

        # Predict the metric value for each player based purely on their team's possession
        predictedY = polynomialEquation(xData)

        # Calculate the residual (Actual Value - Predicted Value)
        # A positive residual means the player overperformed the possession expectation
        # A negative residual means they underperformed it
        residuals = yData - predictedY

        # Replace the original values in the dataframe with the new residual values
        residualDf[column] = residuals

    mergedDf = residualDf.copy()

    tempColumns = finalColumns.copy()
    tempColumns.remove('minutesPerApp')
    tempColumns.remove('currentAge')
    tempColumns.remove('marketValue')
    tempColumns.remove('teamAvgPossession')
    print(len(tempColumns))
    percentileSubset = mergedDf[tempColumns].rank(pct=True, method='min')
    mergedDf[tempColumns] = percentileSubset

    pcaModel = PCA()

    principalComponents = pcaModel.fit_transform(mergedDf[tempColumns])

    pcaColumnNames = [f'PC{i + 1}' for i in range(principalComponents.shape[1])]
    pcaDf = pd.DataFrame(data=principalComponents, columns=pcaColumnNames, index=mergedDf.index)

    explainedVariance = pcaModel.explained_variance_ratio_

    cumulativeVariance = np.cumsum(explainedVariance)

    plt.figure(figsize=(10, 6))

    pcRange = range(1, len(explainedVariance) + 1)
    plt.bar(
        pcRange,
        explainedVariance,
        alpha=0.6,
        align='center',
        label='Individual Explained Variance'
    )

    plt.step(
        pcRange,
        cumulativeVariance,
        where='mid',
        color='red',
        label='Cumulative Explained Variance'
    )

    plt.ylabel('Explained Variance Ratio')
    plt.xlabel('Principal Component Index')
    plt.title('PCA Scree Plot')
    plt.legend(loc='best')
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.show()

    firstFiveLoadings = pcaModel.components_[:5]

    pcNames = ['PC1', 'PC2', 'PC3', 'PC4', 'PC5']

    loadingsDf = pd.DataFrame(
        data=firstFiveLoadings,
        columns=tempColumns,
        index=pcNames
    ).T

    # print(loadingsDf)

    plt.figure(figsize=(10, 12))

    heatmapPlot = sns.heatmap(
        loadingsDf,
        cmap='vlag',
        center=0,
        annot=False,
        cbar_kws={'label': 'Loading Weight'}
    )

    plt.title('PCA Loadings per Component')
    plt.tight_layout()
    plt.show()

    mergedDf[pcNames] = pcaDf[pcNames]

    if displayVersion == "PCA":
        targetDf = mergedDf[
            (mergedDf.index.str.contains(f"{targetPlayer} ", case=False, na=False)) & mergedDf.index.str.contains(
                f"{specificSeason}", case=False, na=False)][
            pcNames]
        print(targetDf)
        targetAvgs = [np.mean(targetDf['PC1']), np.mean(targetDf['PC2']), np.mean(targetDf['PC3']),
                      np.mean(targetDf['PC4']), np.mean(targetDf['PC5'])]

        mergedDf['targetDistance'] = (targetAvgs[0] - mergedDf['PC1']) ** 2 + (targetAvgs[1] - mergedDf['PC2']) ** 2 + (
                targetAvgs[2] - mergedDf['PC3']) ** 2 + (targetAvgs[3] - mergedDf['PC4']) ** 2 + (
                                             targetAvgs[4] - mergedDf['PC5']) ** 2
        distanceDf = mergedDf[
            ['currentAge', 'marketValue', 'targetDistance', 'PC1', 'PC2', 'PC3', 'PC4', 'PC5']].sort_values(
            by='targetDistance', ascending=True)
        individualDf = distanceDf[
            (distanceDf.index.str.contains(yearString, case=False, na=False)) & (distanceDf['currentAge'] <= maxAge) & (
                    distanceDf['marketValue'] <= maxMarketValue)].head(30)
        individualDf[pcNames] = individualDf[pcNames].round(2)
        print(individualDf)
    else:
        targetAvgs = []
        mergedDf['targetDistance'] = 0
        for column in tempColumns:
            targetAvgs.append(np.mean(
                mergedDf[mergedDf.index.str.contains(f"{targetPlayer} {specificSeason}", case=False, na=False)][
                    column]))
            mergedDf['targetDistance'] = mergedDf['targetDistance'] + (np.mean(
                mergedDf[mergedDf.index.str.contains(f"{targetPlayer} {specificSeason}", case=False, na=False)][
                    column]) - mergedDf[column]) ** 2
        distanceDf = mergedDf[['currentAge', 'marketValue', 'targetDistance']].sort_values(by='targetDistance',
                                                                                           ascending=True)
        print(distanceDf[(distanceDf.index.str.contains(yearString, case=False, na=False)) & (
                distanceDf['currentAge'] <= maxAge) & (distanceDf['marketValue'] <= maxMarketValue)].head(30))
        exit()
    if comparisonGraph:
        choice = int(input("What number player would you like to view the comparison graph of? (#) ")) - 1

        mockData = {
            'PC1': [targetAvgs[0], individualDf['PC1'].iloc[choice]],
            'PC2': [targetAvgs[1], individualDf['PC2'].iloc[choice]],
            'PC3': [targetAvgs[2], individualDf['PC3'].iloc[choice]],
            'PC4': [targetAvgs[3], individualDf['PC4'].iloc[choice]],
            'PC5': [targetAvgs[4], individualDf['PC5'].iloc[choice]]
        }
        inputDf = pd.DataFrame(mockData, index=['Rodri', 'comparisonPlayer'])
        print(inputDf)
        # 2. Extract the data for plotting
        numVars = len(inputDf.columns)
        yPositions = np.arange(numVars)

        playerOneData = inputDf.iloc[0].values
        playerTwoData = inputDf.iloc[1].values

        # 3. Set up the figure
        fig, ax = plt.subplots(figsize=(8, 7))

        # 4. Plot the horizontal bars
        # We use standard light blue and green hex codes to match your image
        ax.barh(yPositions, playerOneData, color='green', alpha=0.5, height=0.85, label=inputDf.index[0])
        ax.barh(yPositions, playerTwoData, color='blue', alpha=0.5, height=0.85, label=inputDf.index[1])

        # 5. Add the center dashed zero-line
        ax.axvline(0, color='black', linestyle='--', linewidth=1.5, zorder=3)

        # 6. Formatting to match the minimalist style of the reference image
        ax.set_yticks([])
        ax.set_xticks([])
        ax.invert_yaxis()  # Invert to plot the first column at the top

        # Enforce a solid black border around the entire plot area
        for spine in ax.spines.values():
            spine.set_edgecolor('black')
            spine.set_linewidth(1.5)
            spine.set_visible(True)

        # (Optional) Add a legend if you want to distinguish the colours
        plt.legend(loc='upper right')

        plt.tight_layout()
        plt.show()

searchPosition = "DC"

findSimilarPlayers(searchPosition, jsonRequest(f"/api/v1/search/all?q={"van dijk"}&page=0", 0)['results'][0]['entity']['name'],
                   residualColumnsKey[searchPosition], "25/26", 24, 50000000, "PCA", checkCorrelations=False, specificSeason="",
                   getData=False)
