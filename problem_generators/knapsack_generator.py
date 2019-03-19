def is_positive_integer(X):
    return np.logical_and((X>0),(np.floor(X)==X))

def knapsack(weights, values, W):
    """KNAPSACK Solves the 0-1 knapsack problem for positive integer weights

  [BEST AMOUNT] = KNAPSACK(WEIGHTS, VALUES, CONSTRAINT)
       
       WEIGHTS    : The weight of every item (1-by-N)
       VALUES     : The value of every item (1-by-N)
       CONSTRAINT : The weight constraint of the knapsack (scalar)

       BEST       : Value of best possible knapsack (scalar)
       AMOUNT     : 1-by-N vector specifying the amount to use of each item (0 or 1)


    EXAMPLE :

        weights = [1 1 1 1 2 2 3];
        values  = [1 1 2 3 1 3 5];
        [best amount] = KNAPSACK(weights, values, 7)

        best =

            13


        amount =

             0     0     1     1     0     1     1


   See <a href="http://en.wikipedia.org/wiki/Knapsack_problem">Knapsack problem</a> on Wikipedia.

   Copyright 2009 Petter Strandmark
   <a href="mailto:petter.strandmark@gmail.com">petter.strandmark@gmail.com</a>"""

    if not all(is_positive_integer(weights)) or not is_positive_integer(W):
        raise Exception('Weights must be positive integers');
    
    # We work in one dimension
#     M, N = weights.shape;
    weights = weights[:];
    values = values[:];
    if len(weights) != len(values):
        raise Exception('The size of weights must match the size of values');
    
#     if len(W) > 1:
#         raise Exception('Only one constraint allowed');
      
    
    # Solve the problem
    
    # Note that A would ideally be indexed from A(0..N,0..W) but MATLAB 
    # does not allow this.
    A = np.zeros((len(weights)+1,W+1));
    # A(j+1,Y+1) means the value of the best knapsack with capacity Y using
    # the first j items.
    for j in  range(len(weights)):
        for Y in range(W):
            if weights[j] > Y+1:
                A[j+1,Y+1] = A[j,Y+1];
            else:
                A[j+1,Y+1] = max(A[j,Y+1], values[j] + A[j,Y-weights[j]+1]);
            
        
    

    best = A[-1, -1];
    print(A)
    #Now backtrack 
    amount = np.zeros(len(weights))
    a = best
    j = len(weights)-1
    Y = W-1
    while a > 0:
        while A[j+1,Y+1] == a:
            j = j - 1;
        
        j = j + 1; # This item has to be in the knapsack
        amount[j] = 1;
        Y = Y - weights[j];
        j = j - 1;
        a = A[j+1,Y+1];

    
    # amount = reshape(amount,M,N);
    return best, amount