


"""
script: lab_expression_auto_generator.py

The main goal of this script is to explore the possibility to automatically generate the expression.
In our framework, an expression is equivalent to a trading strategy. The objective is to create an automatic
strategy generator.


Technical Details:

    There are two components in an expression: (1) variable, (2) operator and (3) constant. For example, in the
    expression return + 0.2 * op_mean(adjClose, 5), return and adjClose are variables, * and op_mean are operators,
    0.2 and 5 are constant. Constant are either coefficient or parameters.


    The main idea is to apply genetic algorithm to automatically generate the expressions. In a generic programming, we
    need to define the following operators:
        (1) selection
        (2) crossover
        (3) Mutation


    *** selection ***

    In order to define the selection operator, we should first be able to assign a score to each expression.
    The score should reflect the success/failure of the strategy expression.

    We may use a priority queue to hold the entire population, which is a strategy pool.


    *** crossover ***

    The crossover operator takes two expression and generate a new expression by swapping parts of the two input
    expression. In an idea case, the new expression should be valid.


    *** mutation ***

    With the mutation operator, we should be able to (1) mutate some of the components in the given expression
    (2) add new components to the given expression. The second requirement is to make sure that we can explore
    expression with different number of components.



Implementation Details:

    We will first try to use ast module. The reason is it can parse the expression to a tree structure automatically.



"""

import random

built_in_operators = ['+', '-', '*', '/']


user_defined_operators = [
    'my_op1',
    'my_op2',
    'my_op3',
]


user_defined_variables = [
    'rtn',
    'adjClose',
    'open',
    'high',
    'low',
    'close',
    'volatility',
]

secondary_operator_pool = [
    'my_op1(rtn,1,2)',
    'my_op1(adjClose)',
    'my_op2(open,adjClose)',
    'my_op3(low,2)'
]


expr_1 = 'return + 0.4 * my_op2(adjClose, volatility)'
expr_2 = 'my_op1(adjClose, 3,2) + 0.7 * (open - close)'



"""
we will define two crossover operation:

The first one is to apply crossover operator only on user defined variables

In the above example, in expr_1, we have user defined variables [return, adjClose, volatility] and in expr_2 we have
[adjClose, open, close]. we can apply the crossover operator on the two lists. Assume the pivot index is 2, so we swap
lst1[:2] and lst2[:2]

then we have [adjClose, open, volatility] and [return, adjClose, close] and we get two new expressions:

new_expr_1 = 'adjClose + 0.4 * mp_op2(open, volatility)'
new_expr_2 = 'my_op1(return, 3, 2) + 0.7 * (adjClose - close)'


The second approach will include the user defined operators

Since the different user defined operations have different signature (parameter list), we will consider the user-defined
operator as an entire object.

if we parse the two expressions, we will get
[return, my_op2(adjClose, volatility)]
[my_op1(adjClose, 3, 2), open, close]

if the pivot index  is 1, then we have

[my_op1(adjClose, 3, 2), my_op2(adjClose, volatility)]
[return, open, close]

and the two expressions become

new_expr_1 = 'my_op1(adjClose, 3, 2) + 0.4 * my_op2(adjClose, volatility)'
new_expr_2 = 'return + 0.7 * (open - close)'




"""
import ast
import copy
import numpy as np


expr_1 = 'rtn + 0.4 * my_op2(adjClose, volatility)'
expr_2 = 'my_op1(adjClose, 3,2) + 0.7 * (open - close)'
expr_3 = 'close + my_op1(0.3 * adjClose, my_op2(volatility * 2) - rtn)'



class ExpressionVisitor(ast.NodeVisitor):
    def __init__(self):
        self.list_variables_exclude_operators = []
        self.list_user_defined_operators = []
        self.list_operator_scope = []
        self._expression = []
        self._variable_position = []
        self._number_position = []


    def visit_Name(self, node):
        if node.id in user_defined_variables:
            self.list_variables_exclude_operators.append(node.id)
        self.generic_visit(node)

        self._expression.append(node.id)

        variable_index = len(self.list_variables_exclude_operators) - 1
        pos_in_expression = len(self._expression) - 1

        self._variable_position.append((variable_index, pos_in_expression))

    def visit_Call(self, node):
        operator_name = node.func.id
        if operator_name in user_defined_operators:
            self.list_user_defined_operators.append(operator_name)

        scope_start = len(self._expression)
        operator_index = len(self.list_user_defined_operators) - 1

        # construct the expression

        self._expression.append(operator_name)
        self._expression.append('(')

        # process the args
        for item in node.args:
            self.visit(item)
            self._expression.append(',')


        # process the keywords list
        for item in node.keywords:
            self._expression.append(item.arg)
            self._expression.append('=')
            self.visit(item.value)
            self._expression.append(',')


        # remove the last comma
        self._expression.pop()

        self._expression.append(')')



        scope_end   = len(self._expression)
        self.list_operator_scope.append((operator_index, scope_start, scope_end))

    def visit_Num(self,node):
        self._expression.append(str(node.n))
        self._number_position.append(len(self._expression) - 1)


    def visit_BinOp(self, node):
        self._expression.append('(')
        self.visit(node.left)
        op = node.op

        if isinstance(op, ast.Add):
            self._expression.append('+')
        elif isinstance(op, ast.Sub):
            self._expression.append('-')
        elif isinstance(op, ast.Mult):
            self._expression.append('*')
        elif isinstance(op, ast.Div):
            self._expression.append('/')
        else:
            raise ValueError("Invalid operator.")
        self.visit(node.right)
        self._expression.append(')')




    @property
    def variables(self):
        return copy.deepcopy(self.list_variables_exclude_operators)

    @property
    def operators(self):
        return copy.deepcopy(self.list_user_defined_operators)


    @property
    def operator_scope(self):
        result = copy.deepcopy(self.list_operator_scope)
        result.reverse()
        return result

    @property
    def expression(self):
        return copy.deepcopy(self._expression)

    @property
    def variable_position(self):
        return copy.deepcopy(self._variable_position)


    @property
    def number_position(self):
        return copy.deepcopy(self._number_position)
    
    
    def getStrExpression(self):
        return ''.join(self._expression)





expr_2 = 'my_op1(adjClose, 3,2) + 0.7 * (open - close)'
expr_3 = 'close + my_op1(0.3 * adjClose, my_op2(volatility * 2) - rtn)'


expr2_visitor = ExpressionVisitor()
expr3_visitor = ExpressionVisitor()

expr2_visitor.visit(ast.parse(expr_2))
expr3_visitor.visit(ast.parse(expr_3))

print (expr2_visitor.variables)
print (expr2_visitor.operators)
print (expr2_visitor.operator_scope)
print (expr2_visitor.expression)
print (expr2_visitor.variable_position)
print (expr2_visitor.getStrExpression())



print (expr3_visitor.variables)
print (expr3_visitor.operators)
print (expr3_visitor.operator_scope)
print (expr3_visitor.expression)






def construct_new_variables_list(variable_position, operator_scope):
    i_variable    = 0
    i_operator    = 0
    new_variables = []
    end_operator_scope = 0

    while i_variable < len(variable_position) and i_operator < len(operator_scope):
        #print ' '.join([str(x) for x in [i_variable, i_operator, variable_position[i_variable][1],operator_scope[i_operator][1], end_operator_scope]])
        _variable_position = variable_position[i_variable][1]
        if _variable_position < operator_scope[i_operator][1] and variable_position >= end_operator_scope:
            new_variables.append(variable_position[i_variable])
            i_variable += 1
        elif _variable_position < operator_scope[i_operator][1]:
            i_variable += 1
        else:
            # In this case, we need to add operator to the new variables list
            # and update teh end_operator_scope variable
            new_variables.append(operator_scope[i_operator])
            end_operator_scope = operator_scope[i_operator][2]
            i_operator += 1


    for i in xrange(i_variable, len(variable_position)):
        if variable_position[i][1] >= end_operator_scope:
            new_variables.append(variable_position[i])

    for i in xrange(i_operator, len(operator_scope)):
        new_variables.append(operator_scope[i])


    return new_variables



"""  ===== Test ======

variable_position = expr2_visitor.variable_position
operator_scope    = expr2_visitor.operator_scope

print variable_position
print operator_scope


new_variables = construct_new_variables_list(variable_position, operator_scope)
print new_variables


"""





def crossover(expr1, expr2, p_opSwap=0.05):
    """
    This function implements the crossover operator in genetic algorithm.

    Args:
        expr1, expr2: string expression.
        p_opSwap:     the probability that a operator swap happens. The default value is 0.05
    """
    visitor1 = ExpressionVisitor()
    visitor2 = ExpressionVisitor()
    visitor1.visit(ast.parse(expr1))
    visitor2.visit(ast.parse(expr2))

    variables_1         = visitor1.variables
    variable_position_1 = visitor1.variable_position
    operator_scope_1    = visitor1.operator_scope
    expression_1        = visitor1.expression


    variables_2         = visitor2.variables
    variable_position_2 = visitor2.variable_position
    operator_scope_2    = visitor2.operator_scope
    expression_2        = visitor2.expression


    rand = np.random.uniform()

    if rand > p_opSwap:
        # we only swap user defined variables in the expression
        k = np.random.randint(min(len(variables_1), len(variables_2)))
        for i in range(k + 1):
            expression_1[variable_position_1[i][1]] = variables_2[i]
            expression_2[variable_position_2[i][1]] = variables_1[i]

    else:
        # consider user-defined operators as a single variable

        # here we need to construct a new variable list
        new_variables_1 = construct_new_variables_list(variable_position_1, operator_scope_1)
        new_variables_2 = construct_new_variables_list(variable_position_2, operator_scope_2)

        k = np.random.randint(min(len(new_variables_1), len(new_variables_2)))

        for i in range(k+1):
            item1 = new_variables_1[i]
            item2 = new_variables_2[i]

            # there are three situations

            # (1) swap variable and variables
            if len(item1) == len(item2) == 2:
                expression_1[item1[1]] = variables_2[item2[0]]
                expression_2[item2[1]] = variables_1[item1[0]]

            # (2) swap variable and operator
            elif len(item1) == 2 and len(item2) == 3:
                sub_expression = []
                for k in range(item2[1], item2[2]):
                    sub_expression.append(expression_2[k])

                expression_2[item2[1]] = variables_1[item1[0]]
                for k in range(item2[1]+1, item2[2]):
                    expression_2[k] = ''

                expression_1[item1[1]] = ''.join(sub_expression)

            elif len(item1) == 3 and len(item2) == 2:
                sub_expression = []
                for k in range(item1[1], item1[2]):
                    sub_expression.append(expression_1[k])

                expression_1[item1[1]] = variables_2[item2[0]]
                for k in range(item1[1] + 1, item1[2]):
                    expression_1[k] = ''

                expression_2[item2[1]] = ''.join(sub_expression)

            # (3) swap operator and operator
            else:
                sub_expression_1 = []
                sub_expression_2 = []
                for k in range(item1[1], item1[2]):
                    sub_expression_1.append(expression_1[k])

                for k in range(item2[1], item2[2]):
                    sub_expression_2.append(expression_2[k])

                expression_1[item1[1]] = ''.join(sub_expression_2)
                expression_2[item2[1]] = ''.join(sub_expression_1)

                for k in range(item1[1]+1, item1[2]):
                    expression_1[k] = ''

                for k in range(item2[1]+1, item2[2]):
                    expression_2[k] = ''

    return (''.join(expression_1), ''.join(expression_2))



"""
========== Test ================

expr_2 = 'my_op1(adjClose, 3,2) + 0.7 * (open - close)'
expr_3 = 'close + my_op1(0.3 * adjClose, my_op2(volatility * 2) - rtn)'

print (crossover(expr_2, expr_3))

"""









expr_2 = 'my_op1(adjClose, 3,2) + 0.7 * (open - close)'
expr_3 = 'close + my_op1(0.3 * adjClose, my_op2(volatility * 2) - rtn)'


expr2_visitor = ExpressionVisitor()
expr3_visitor = ExpressionVisitor()

expr2_visitor.visit(ast.parse(expr_2))
expr3_visitor.visit(ast.parse(expr_3))

print (expr2_visitor.variables)
print (expr2_visitor.operators)
print (expr2_visitor.operator_scope)
print (expr2_visitor.expression)
print (expr2_visitor.variable_position)
print (expr2_visitor.getStrExpression())



print (expr3_visitor.variables)
print (expr3_visitor.operators)
print (expr3_visitor.operator_scope)
print (expr3_visitor.expression)



"""
In this section, we will implementat the mutate operator.

Like crossover operator, here we also have two possibility:
    (1) apply mutate operator only on user-defined variables or built-in operators
    (2) add user-defined operators to the end of the expression

Note that in the current implementation, we do not mutate the user-defined operators. The reasons is the signature
of user-defined operators are quite different case by case. If we mutatet the user-defined operators, it is hard to
guarantee the new operator is a valid operators.

"""



def mutate(expr, p_var=0.02, p_op=0.02, p_num=0.05, p_div=0.02, p_addOp=0.05):
    visitor = ExpressionVisitor()
    visitor.visit(ast.parse(expr))

    variables         = visitor.variables
    variable_position = visitor.variable_position
    expression        = visitor.expression
    number_position   = visitor.number_position

    rand_var = np.random.uniform()


    # mutate the user-defined variables
    rand_var = np.random.uniform(size=len(variable_position))
    for k, item in enumerate(variable_position):
        if rand_var[k] < p_var:  # do a mutation
            expression[item[1]] = random.choice(user_defined_variables)


    # mutate the built-in operators
    for i in xrange(len(expression)):
        if expression[i] in built_in_operators:
            rand_op = np.random.uniform()
            if rand_op < p_op:
                new_op = random.choice(built_in_operators)
                if new_op == '/':
                    # this part is to reduce the appearance of the division operator.
                    # because if the new operator is '/', it is not guaranteed that it can be compiled
                    expression[i] = new_op if np.random.uniform() < p_div else expression[i]
                else:
                    expression[i] = new_op


    # mutate the numbers
    rand_num = np.random.uniform(size=len(number_position))
    rand_normal = np.random.normal(0, 0.05, len(rand_num))

    for k, item in enumerate(number_position):
        if rand_num[k] < p_num:     # mutation on numbers
            n = float(expression[item]) * (1. + rand_normal[k])
            expression[item] = str(n)


    # mutation: if we add more user-defined operators
    rand_addOp = np.random.uniform()
    if rand_addOp < p_addOp:
        new_subexpression = ''.join([ '+', '1.0', '*' , random.choice(secondary_operator_pool)])
        expression.append(new_subexpression)

    return ''.join(expression)



# ==== Test =====
expr_3 = 'close + my_op1(0.3 * adjClose, my_op2(volatility * 2) - rtn)'
visitor = ExpressionVisitor()
visitor.visit(ast.parse(expr_3))
print visitor.number_position

for i in xrange(100):
    print  mutate(expr_3, p_addOp=0.03)


