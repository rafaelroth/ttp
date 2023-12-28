import time
import numpy as np
import pandas as pd
from geopy.distance import geodesic
from io import StringIO
import random
import math




maxP = 500
maxC = 250
maxR = 5
T = float(400)
beta = 0.9999
weight = 4000
teta = 1.04

data = """
Time,Cidade,Latitude,Longitude
América-MG,Belo Horizonte-MG,-19.9191,-43.9386
Athletico-PR,Curitiba-PR,-25.4290,-49.2671
Atlético-MG,Belo Horizonte-MG,-19.9191,-43.9386
Bahia,Salvador-BA,-12.9718,-38.5016
Botafogo,Rio de Janeiro-RJ,-22.9068,-43.1729
Corinthians,São Paulo-SP,-23.5505,-46.6333
Coritiba,Curitiba-PR,-25.4290,-49.2671
Cruzeiro,Belo Horizonte-MG,-19.9191,-43.9386
Cuiabá,Cuiabá-MT,-15.6014,-56.0979
Flamengo,Rio de Janeiro-RJ,-22.9068,-43.1729
Fluminense,Rio de Janeiro-RJ,-22.9068,-43.1729
Fortaleza,Fortaleza-CE,-3.7319,-38.5267
Goiás,Goiânia-GO,-16.6799,-49.2550
Grêmio,Porto Alegre-RS,-30.0346,-51.2177
Internacional,Porto Alegre-RS,-30.0346,-51.2177
Palmeiras,São Paulo-SP,-23.5505,-46.6333
Bragantino,Bragança Paulista-SP,-22.9519,-46.5419
Santos,Santos-SP,-23.9608,-46.3336
São Paulo,São Paulo-SP,-23.5505,-46.6333
Vasco da Gama,Rio de Janeiro-RJ,-22.9068,-43.1729
"""

data = StringIO(data)
teams = pd.read_csv(data, index_col="Time")


def teams_distance(team_from: str, team_to: str = "Cruzeiro", teams: pd.DataFrame = teams)->float:
    lat_long = ["Latitude", "Longitude"]

    origin = teams.loc[team_from, lat_long].agg(tuple)
    destination = teams.loc[team_to, lat_long].agg(tuple)

    return round(geodesic(origin, destination).km * 1.60934, 2)


def create_adjacency_matrix(teams: pd.DataFrame) -> pd.DataFrame:
    team_names = teams.index
    adjacency_matrix = pd.DataFrame(columns=team_names, index=team_names)

    for i, team_from in enumerate(team_names):
        for team_to in team_names[i:]:
            distance = teams_distance(team_from, team_to, teams)
            adjacency_matrix.loc[team_from, team_to] = distance
            adjacency_matrix.loc[team_to, team_from] = distance

    return adjacency_matrix



distance_matrix = create_adjacency_matrix(teams)

# Number of teams will be length of distance_matrix            
numberOfTeams = int(len(distance_matrix))
halfTeams = int(numberOfTeams/2)

# Defines number of rounds
numberOfRounds = int(2*(numberOfTeams - 1))
halfRounds = int(numberOfRounds/2)

# Converts array into a numpy array for faster computation. And changes type of variable to float.
distance_matrix = np.array(distance_matrix, dtype=float)

"""
Part 2
"""


def findCanonicalPattern():
    x = halfRounds
    y = halfTeams
    z = 2
    E = np.zeros((x,y,z)) 
    
    for i in range(halfRounds):
        
        E[i][0][:]=[numberOfTeams,i + 1]        # The first edge of a round is the last team (e.g. team 4) playing team i + 1 
        
        for k in range(halfTeams-1):      # Then to fill the last edges, use functions F1 and F2
            
            E[i][k+1][:]=[F1(i + 1, k + 1, numberOfTeams), F2(i + 1, k + 1, numberOfTeams)] 
         
    return(E) 


def F1(i,k,numberOfTeams):
    if i + k < numberOfTeams:
        return(i + k)
    else:
        
        return(i + k - numberOfTeams + 1)
    
    

def F2(i,k,numberOfTeams):
    
    if i - k > 0:
        
        return(i - k)
        
    else:
        
        return(i - k + numberOfTeams - 1)
        


def getInitialSolution(numberOfTeams):
    solution = np.zeros((numberOfTeams,numberOfRounds), dtype=int)

    games = findCanonicalPattern()
    

    for i in range(halfRounds):
        for k in range(halfTeams):
            edge = games[i][k]
            teamA = int(edge[0])
            teamB = int(edge[1])

            solution[teamA - 1][i] = teamB
            solution[teamB - 1][i] = - teamA

    temp = solution.copy()
    temp = -1*np.roll(temp, halfRounds, axis=1)
    solution = solution+temp

    return(solution)


def simulated_annealing(maxP,maxC,maxR,T,beta,weight,teta):

    S = getInitialSolution(numberOfTeams)
    S = np.array(S)

    numberOfViolations = 0

    for i in range(numberOfTeams):
        count = 0
        for k in range(1,numberOfRounds):
            if (S[i][k] > 0 and S[i][k-1] > 0) or (S[i][k] < 0 and S[i][k-1] < 0):
                count += 1
            else:
                count = 0
            
            # If count is bigger than 2, you're playing more than three consecutive games away or at home. So increments number of violations by 1.
            if count > 2:

                numberOfViolations += 1
            
            # If you're playing the same team in two consecutive rounds, increment number of violations by 1.
            if abs(S[i][k]) == abs(S[i][k-1]):

                numberOfViolations += 1
    
    violationsS = numberOfViolations
    totaldistance = 0

    distanceS = np.copy(S)
                
    for x in range(numberOfTeams):
        distanceS[x] = [x + 1 if i > 0 else abs(i) for i in distanceS[x]]
        totaldistance += distance_matrix[x][distanceS[x][0]-1]
    
        for y in range(numberOfRounds - 1):
            totaldistance += distance_matrix[distanceS[x][y]-1][distanceS[x][y + 1]-1]

        totaldistance += distance_matrix[distanceS[x][-1] - 1][x]
                    
    costS = totaldistance

    if violationsS != 0:
        costS = math.sqrt((costS**2) + (weight*(1 + math.sqrt(violationsS)*math.log(violationsS/float(2))))**2)

    bestFeasible = 9999999
    nbf = 9999999
    bestInfeasible = 9999999
    nbi = 9999999
    reheat = 0
    counter = 0

    bestSolution = None
    

    while reheat <= maxR:
        
        phase = 0
        
        # While system has not decreased the temperature maxP times without improving the solution
        while phase <= maxP:
            counter = 0

            # While system has not rejected maxC moves
            while counter <= maxC:

                # Choose random move
                chooseMove = random.randint(0,4)
                if chooseMove == 0:     # Swap Homes
                    newS = np.copy(S)   # The new solution is a copy of the current one

                    teamA = random.randint(0,numberOfTeams - 1)         # Choose random team A
                    teamB = random.randint(0,numberOfTeams - 1)         # Choose random team B
                    
                    for i in range(numberOfRounds):
                        # When team A and team B play each other
                        if abs(S[teamA][i]) == teamB + 1:
                            # Invert the signs (swap homes)
                            newS[teamA][i] = - S[teamA][i]
                            newS[teamB][i] = - S[teamB][i]
    
                
                elif chooseMove == 1:   # Swap Rounds
                    newS = np.copy(S)

                    roundAindex = random.randint(0,numberOfRounds - 1)  # Choose random round A
                    roundBindex = random.randint(0,numberOfRounds - 1)  # Choose random round B
                    
                    # Swap columns of array (swap rounds)
                    newS[:,[roundAindex,roundBindex]] =  newS[:,[roundBindex,roundAindex]]
                    
                    
                elif chooseMove == 2:   # Swap Teams
                    newS = np.copy(S)

                    teamA = random.randint(0,numberOfTeams - 1)         # Choose random team A
                    teamB = random.randint(0,numberOfTeams - 1)         # Choose random team B
                    for i in range(numberOfRounds):
                        if abs(S[teamA][i]) != teamB + 1:               # If team A and team B are not playing each other in round i

                            newS[[teamA,teamB],i] =  newS[[teamB,teamA],i]      # Swap their values (swap their opponents)
                            
                            formerAdversaryTeamA = abs(S[teamA][i]) - 1         # Gets former opponent of team A
                            formerAdversaryTeamB = abs(S[teamB][i]) - 1         # Gets former opponent of team B
                            
                            # Now the former opponent of team A plays B at home or away
                            if S[formerAdversaryTeamA][i] > 0:
                                newS[formerAdversaryTeamA][i] = teamB + 1       
            
                            else:
                                newS[formerAdversaryTeamA][i] = -(teamB + 1)
                            
                            # Now the former opponent of team B plays A at home or away
                            if S[formerAdversaryTeamB][i] > 0:
                                newS[formerAdversaryTeamB][i] = teamA + 1
            
                            else:
                                newS[formerAdversaryTeamB][i] = -(teamA + 1)
                    
                
                elif chooseMove == 3:   # Partial Swap Rounds
                    newS = np.copy(S)
                    
                    team = random.randint(0,numberOfTeams - 1)          # Choose random team
                    roundAindex = random.randint(0,numberOfRounds - 1)  # Choose random round A
                    roundBindex = random.randint(0,numberOfRounds - 1)  # Choose random round B
                    
                    startCircuit = abs(S[team][roundAindex])
                    finishCircuit = abs(S[team][roundBindex])
      
                    currentTeam = startCircuit
                    currentRound = roundBindex
                    
                    # Swaps values of Round A and B for the chosen team
                    newS[team,[roundAindex,roundBindex]] =  newS[team,[roundBindex,roundAindex]]
                    
                    # Now you must figure out the other teams that you have to swap to fix the schedule, and swap their values
                    while currentTeam != finishCircuit:
                        
                        index = currentTeam - 1
                        
                        newS[index,[roundAindex,roundBindex]] =  newS[index,[roundBindex,roundAindex]]
                        
                        currentTeam = abs(S[currentTeam - 1][currentRound])
        
                        if currentRound == roundBindex:
            
                            currentRound = roundAindex
            
                        else:
            
                            currentRound = roundBindex
                    
                    index = currentTeam - 1
                    
                    newS[index,[roundAindex,roundBindex]] =  newS[index,[roundBindex,roundAindex]]
                    
                    
                elif chooseMove == 4:   # Partial Swap Teams
                    newS = np.copy(S)
                    
                    round = random.randint(0,numberOfRounds - 1)
                    teamA = random.randint(0,numberOfTeams - 1)
                    teamB = random.randint(0,numberOfTeams - 1)
                    
                    adversaryA = S[teamA][round]
                    adversaryB = S[teamB][round]
                    
                    # If team A and B are not playing each other, execute the swap
                    if abs(adversaryB) != teamA + 1:
                        
                        # Swap the teams in the selected round
                        newS[[teamA,teamB],round] =  newS[[teamB,teamA],round]
                        
                        affectedTeamA = abs(adversaryA)
                        affectedTeamB = abs(adversaryB)
                            
                        oppositeA = S[affectedTeamA - 1][round]
                        oppositeB = S[affectedTeamB - 1][round]
                        
                        # Fix the problem you created (e.g. the opponent of A now plays B)
                        if oppositeA > 0:
                                
                            newS[affectedTeamA - 1][round] = abs(oppositeB)
            
                        else:
                                
                            newS[affectedTeamA - 1][round] = - abs(oppositeB)
            
                        if oppositeB > 0:
                                
                            newS[affectedTeamB - 1][round] = abs(oppositeA)
                
                        else:
                                
                            newS[affectedTeamB - 1][round] = - abs(oppositeA)
                        
                        
                        currentAdversaryB = adversaryB
                        
                        # Look for problems you generated thoroughout the schedule, and swap them as well
                        while currentAdversaryB != adversaryA:
                            currentAdversaryA = currentAdversaryB
                            
                            # e.g. after doing the first swap, now team A plays team 6 at home twice. So you have to find where A played 6 at home before and swap that entry.
                            i = np.nonzero(S[teamA] == currentAdversaryA)[0][0]
                                    
                            currentAdversaryB = S[teamB][i]
                            
                            newS[[teamA,teamB],i] =  newS[[teamB,teamA],i]
                        
                            affectedTeamA = abs(currentAdversaryA)
                            affectedTeamB = abs(currentAdversaryB)
                            
                            oppositeA = S[affectedTeamA - 1][i]
                            oppositeB = S[affectedTeamB - 1][i]
                            
                            # Fix the problem you created (e.g. the opponent of A now plays B)
                            if oppositeA > 0:
                                
                                newS[affectedTeamA - 1][i] = abs(oppositeB)
            
                            else:
                                
                                newS[affectedTeamA - 1][i] = - abs(oppositeB)
            
                            if oppositeB > 0:
                                
                                newS[affectedTeamB - 1][i] = abs(oppositeA)
                
                            else:
                                
                                newS[affectedTeamB - 1][i] = - abs(oppositeA)
                                    
                
                # Now that you have a new solution, get the number of violations
                numberOfViolations = 0

                for i in range(numberOfTeams):
                    count = 0

                    for k in range(1,numberOfRounds):
        
                        if (newS[i][k] > 0 and newS[i][k-1] > 0) or (newS[i][k] < 0 and newS[i][k-1] < 0):

                            count += 1
            
                        else:

                            count = 0
            
                        if count > 2:

                            numberOfViolations += 1
            
                        if abs(newS[i][k]) == abs(newS[i][k-1]):

                            numberOfViolations += 1
                
                
                violationsNewS = numberOfViolations

                totaldistance = 0
                distanceNewS = np.copy(newS)
                
                for x in range(numberOfTeams):
                    distanceNewS[x] = [x + 1 if i > 0 else abs(i) for i in distanceNewS[x]]
    
                    totaldistance += distance_matrix[x][distanceNewS[x][0]-1]
    
                    for y in range(numberOfRounds - 1):
        
                        totaldistance += distance_matrix[distanceNewS[x][y]-1][distanceNewS[x][y + 1]-1]
        
                    totaldistance += distance_matrix[distanceNewS[x][-1] - 1][x]
                    
                costNewS = totaldistance
                
                # If the solution is infeasible, penalize it.
                if violationsNewS != 0:

                    costNewS = math.sqrt((costNewS**2) + (weight*(1 + math.sqrt(violationsNewS)*math.log(violationsNewS/float(2))))**2)
                
                # If the new solution improves the current solution, or the best feasible solution so far, or the best infeasible solution so far
                if costNewS < costS or (violationsNewS == 0 and costNewS < bestFeasible) or (violationsNewS > 0 and costNewS < bestInfeasible):

                    accept = 1
                    
                # Else, accept with a probability given by exp(-delta/T)    
                else:

                    delta = float(costNewS - costS)
                    probability = math.exp(-(delta/T))
                    chance = random.random()
                
                    if chance < probability:

                        accept = 1

                    else:

                        accept = 0
                    
                # If you accepted the solution
                if accept == 1:
                    S = newS
                    
                    violationsS = violationsNewS
                    
                    costS = costNewS
                    
                    # If the new solution is feasible
                    if violationsS == 0:
                        if costS < bestFeasible:
                            bestFeasible = costS
                            bestSolution = S.copy()  # Armazena a melhor solução
                        # The new best feasible will be the minimum between the new solution and the best feasible
                        nbf = min(costS, bestFeasible)
                    
                    else:
                        
                        nbi = min(costS, bestInfeasible)
                    
                    # If the new solution is the best feasible or the best infeasible
                    if nbf < bestFeasible or nbi < bestInfeasible:

                        bestTime = time.process_time()
                        print("Cost = "+str(nbf)+" at "+str(bestTime)+" seconds")     # Print the cost and the time
                        
                        # Reset variables to zero.
                        reheat = 0
                        counter = 0
                        phase = 0
                        
                        # Best temperature will be the current temperature. So when the system reheats, the temperature will be 2*bestTemperature
                        bestTemperature = T
                        
                        # Update best feasible and infeasible costs
                        bestFeasible = nbf
                        bestInfeasible = nbi
                        
                        # Strategic oscilation:
                        # If the new solution is feasible, decrease the weight
                        if violationsS == 0:

                            weight = weight/teta
                        
                        # Else, increase it.
                        else:

                            weight = weight*teta
              
                
                # If you rejected the move, increment counter.
                else:
                        
                    counter += 1
                
            # If counter exceeded maxC, increment phase    
            phase += 1
            
            # and decrease temperature
            T = T*beta
            
            print("Cooling "+str(phase)+" T = "+str(T)+" Cost = "+str(nbf)+" at "+str(bestTime)+" seconds")
        
        
        # When phase exceeds maxP, reheat the system bringing its temperature back to two times the current best temperature
        reheat += 1
        
        T = 2*bestTemperature
        
        print("Reheating")

    if bestSolution is not None:
        print("Melhor solução encontrada:")
        print(bestSolution)
        return bestSolution
    else:
        print("Nenhuma solução factível encontrada.")


def games_by_round(schedule):
    num_teams = schedule.shape[0]
    num_rounds = schedule.shape[1]
    games = [[] for _ in range(num_rounds)]

    for round_index in range(num_rounds):
        for team in range(1, num_teams + 1):
            if schedule[team - 1, round_index] > 0:
                games[round_index].append((team, schedule[team - 1, round_index]))

    return games


def print_schedule(schedule):
    games = games_by_round(schedule)
    team_names = teams.index.tolist()

    for round_index, round_games in enumerate(games):
        print(f"Rodada {round_index + 1}:")
        for home, away in round_games:
            home_team = team_names[home - 1]
            away_team = team_names[abs(away) - 1]
            print(f"{home_team} x {away_team}")
        print()  # Nova linha para separar as rodadas


def total_distance_traveled(schedule, distance_matrix):
    totaldistance = 0

    distanceS = np.copy(schedule)

    team_names = teams.index.tolist()
    distances = {team: 0 for team in team_names}
    for x in range(numberOfTeams):
        team_name = team_names[x]
        # If the entry is positive, team x is playing at home. So make the entry equal to team x. Else, take the absolute of the entry in the schedule.
        distanceS[x] = [x + 1 if i > 0 else abs(i) for i in distanceS[x]]

        # Starts by adding distance from team x to first entry in the schedule. If first entry is team x, you're playing at home and the distance added is 0.
        distances[team_name] += distance_matrix[x][distanceS[x][0]-1]

        for y in range(numberOfRounds - 1):
            # Then it adds distance from entry 1 to entry 2, entry 2 to entry 3, and so on.
            distances[team_name] += distance_matrix[distanceS[x][y]-1][distanceS[x][y + 1]-1]

        # Finally, it adds distance from last entry to team x. If you're playing the last game at home, the distance added is 0.
        distances[team_name] += distance_matrix[distanceS[x][-1] - 1][x]

    for team_name, distance in distances.items():
        totaldistance += distance
        print(f"{team_name}: {distance:.2f} km")

    print(f"TOTAL PERCORRIDO: {totaldistance: .2f} km")



def to_np_array(mlist):
    return [np.array(sublist) for sublist in mlist]


if __name__ == '__main__':
    # Starts clock
    time.process_time()
    # Calls function
    schedule = simulated_annealing(maxP,maxC,maxR,T,beta,weight,teta)
    print_schedule(schedule)
    total_distance_traveled(schedule, distance_matrix)



