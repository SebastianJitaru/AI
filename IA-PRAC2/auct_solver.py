import msat_runner
import wcnf
import argparse

def parse_command_line_arguments(argv=None):
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("solver", help="Path to the MaxSAT solver.")
    parser.add_argument("auction", help="Path to the file that descrives the"
                                      " input auction.")
    parser.add_argument("--no-min-win-bids",
                        help="optional ffag bids",action="store_false", default=True)
    return parser.parse_args(args=argv)
    
class Auction:
    def __init__(self):
        self.bids = dict()
        self.agents = []
        self.formula = wcnf.WCNFFormula()
        self.goods = []
        self.args = parse_command_line_arguments()
        self.solver = msat_runner.MaxSATRunner(self.args.solver)

        self.readfile()
        self.SoftClauses()
        self.HardClauses()
        if self.args.no_min_win_bids:
            self.noMinWinBid()
        self.printResult()

    def readfile(self):
        with open(self.args.auction) as f:
            self.agents = f.readline()[1:].split()
            self.goods = f.readline()[1:].split()
            self.bids = dict()
            for i, line in enumerate(f,1):
                self.bids[i] = line.split()
            self.formula.extend_vars(len(self.bids))
    
    def SoftClauses(self):
        for key in self.bids:
            bid = self.bids[key]
            self.formula.add_clause([key],weight=int(bid[-1]))
    
    def HardClauses(self):
        for i in self.bids:
            bid1 = self.bids[i]
            for j in self.bids:
                bid2 = self.bids[j]
                if(any(items in bid1[1:-1] for items in bid2[1:-1])  and i < j):
                    self.formula.add_clause([-i,-j])
    def noMinWinBid(self):
        for agent in self.agents:
            list=[]
            for i in self.bids:
                bid = self.bids[i]
                if agent == bid[0]:
                    list.append(i)
            self.formula.add_clause(list)
    
    def checkIfValidSolution(self):
        _, model = self.solver.solve(self.formula)
        winningBids = [n for n in model if n > 0]
        winningAgents = []
        for key, value in self.bids.items():
            if(winningBids.__contains__(key)):
                winningAgents.append(value[0])
        result =  all(elem in self.agents for elem in winningAgents)
        if result:
            print("Valid")

    def printResult(self):
        _, model = self.solver.solve(self.formula)
        selectedAuct = [n for n in model if n > 0]
        benefit=0
        
        print("Bids: [")
        for key, value in self.bids.items():
            print("     ",value[0],":",*value[1:-1],"(Price",value[-1],")")
        print("]")
        #print(solver.solve(self.formula))
        if model:    
            for key, value in self.bids.items():
                if(selectedAuct.__contains__(key)):
                    print(value[0],":",*value[1:-1],"(Price",value[-1],")")
                    benefit+=int(value[-1])
            print(f"Benefit:", benefit)   
            self.checkIfValidSolution()
            
        else:
            print("Unsatisfiable")

if __name__ == "__main__":
     auction = Auction()
     #print(auction.formula)