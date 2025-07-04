const std = @import("std");

/// An abstract syntax tree node. Main bulk of the
/// AST object.
pub const ASTNode = union(enum) {
    /// Constants are allowed to be numeric values only.
    /// i: i64 - A signed integer.
    /// u: u64 - An unsigned integer.
    /// f: f64 - A float.
    const Constant = union(enum) {
        i: i64,
        u: u64,
        f: f64,
    };

    /// Indexing requires a pointer variable and an index
    const Index = struct {
        pointer: *ASTNode,
        index: *ASTNode,
    };

    /// Assignments need a left hand side and a right hand side.
    const Assignment = struct {
        lhs: *ASTNode,
        rhs: *ASTNode,
    };

    /// Binary operations need a type signature, a left hand side, and a right hand side (potentially wrapped in parentheses).
    const BinaryOp = struct {
        op_type: enum { lt, lte, gt, gte, eq, neq, add, sub, mul, div, rem, land, lor },
        parentheses: bool = false,
        lhs: *ASTNode,
        rhs: *ASTNode,
    };

    /// Statements need an optional return type and an expresion.
    const Statement = struct {
        return_type: ?enum { int, uint, float, int_p, uint_p, float_p } = null,
        expression: *ASTNode,
    };

    /// Branching involves a condition, an if block, and an optional else block.
    const Branch = struct {
        condition: *ASTNode,
        if_body: *ASTNode,
        else_body: ?*ASTNode = null,
    };

    /// While loops include a condition and a block of statements.
    const WhileLoop = struct {
        condition: *ASTNode,
        body: *ASTNode,
    };

    /// Invoke includes a function name and a list of arguments
    const Invoke = struct {
        name: []const u8,
        args: []const *ASTNode,
    };

    /// constant: Constant - A number constant.
    /// variable: []const u8 - A variable with the name variable.
    /// index: Index - Indexing into a pointer variable.
    /// assignment - An assignment (a = 0).
    /// binary_op - A binary operation (i <= 10).
    /// statement - A statement (int a = 5;).
    /// branch - An if-else statement.
    /// while_loop - A while loop.
    /// block - A series of statements and expressions.
    /// invoke - Invoke a function.
    constant: Constant,
    variable: []const u8,
    index: Index,
    assignment: Assignment,
    binary_op: BinaryOp,
    statement: Statement,
    branch: Branch,
    while_loop: WhileLoop,
    block: std.ArrayList(*ASTNode),
    invoke: Invoke,

    const Self = @This();

    /// Destroy a ASTNode and it's children
    /// Arguments:
    ///     allocator: std.mem.Allocator - The allocator used to initialize the node.
    pub fn deinit(self: *Self, allocator: std.mem.Allocator) void {
        switch (self.*) {
            .constant, .variable => {
                allocator.destroy(self);
            },
            .index => |*index| {
                index.pointer.deinit(allocator);
                index.index.deinit(allocator);
                allocator.destroy(self);
            },
            .assignment => |*assignment| {
                assignment.lhs.deinit(allocator);
                assignment.rhs.deinit(allocator);
                allocator.destroy(self);
            },
            .binary_op => |*binary_op| {
                binary_op.lhs.deinit(allocator);
                binary_op.rhs.deinit(allocator);
                allocator.destroy(self);
            },
            .statement => |*statement| {
                statement.expression.deinit(allocator);
                allocator.destroy(self);
            },
            .branch => |*branch| {
                branch.condition.deinit(allocator);
                branch.if_body.deinit(allocator);
                if (branch.else_body) |*else_body| {
                    else_body.*.deinit(allocator);
                }
                allocator.destroy(self);
            },
            .while_loop => |*while_loop| {
                while_loop.condition.deinit(allocator);
                while_loop.body.deinit(allocator);
                allocator.destroy(self);
            },
            .block => |*block| {
                for (0..block.items.len) |i| {
                    block.items[i].deinit(allocator);
                }
                allocator.destroy(self);
            },
            .invoke => |*invoke| {
                for (0..invoke.args.len) |i| {
                    invoke.args[i].deinit(allocator);
                }
                allocator.destroy(self);
            },
        }
    }

    /// Create a constant node type.
    /// Arguments:
    ///     value: ConstantType - A value to fill the constant.
    /// Returns:
    ///     An instantiated constant.
    pub fn create_constant(comptime T: type, value: T) Self {
        switch (@typeInfo(T)) {
            .int => |i| {
                if (i.signedness == .signed) {
                    return Self{ .constant = .{ .i = value } };
                } else {
                    return Self{ .constant = .{ .u = value } };
                }
            },
            .float => |_| {
                return Self{ .constant = .{ .f = value } };
            },
            else => unreachable,
        }
    }

    /// Create a variable node type.
    /// Arguments:
    ///     name: []const u8 - A name to give the variable.
    /// Returns:
    ///     An instantiated variable.
    pub fn create_variable(name: []const u8) Self {
        return Self{ .variable = name };
    }

    /// Create an index node type.
    /// Arguments:
    ///     index: Index - A pointer and index value for the index.
    /// Returns:
    ///     An instantiated index.
    pub fn create_index(index: Index) Self {
        return Self{ .index = index };
    }

    /// Create an assignment node type.
    /// Arguments:
    ///     assignment: Assignment - The left and right hand side of the assignment.
    /// Returns:
    ///     An instantiated assignment.
    pub fn create_assignment(assignment: Assignment) Self {
        return Self{ .assignment = assignment };
    }

    /// Create a binary operation.
    /// Arguments:
    ///     binary_op: BinaryOp - The operation type, left hand side, and right hand side of the operation.
    /// Returns:
    ///     An intantiated binary operation.
    pub fn create_binary_op(binary_op: BinaryOp) Self {
        return Self{ .binary_op = binary_op };
    }

    /// Create a statement node type.
    /// Arguments:
    ///     statement: Statement -The declaration type and expression of the statement.
    /// Returns:
    ///     An instantiated statement.
    pub fn create_statement(statement: Statement) Self {
        return Self{ .statement = statement };
    }

    /// Create a branch statement.
    /// Arguments:
    ///     branch: Branch - The condition, if block, and else block of the branch statement.
    /// Returns:
    ///     An instantiated branch statement.
    pub fn create_branch(branch: Branch) Self {
        return Self{ .branch = branch };
    }

    /// Create a while loop.
    /// Arguments:
    ///     while_loop: WhileLoop - The condition and body of the while loop.
    /// Returns:
    ///     An instantiated while loop.
    pub fn create_while_loop(while_loop: WhileLoop) Self {
        return Self{ .while_loop = while_loop };
    }

    /// Instantiate a block.
    /// Arguments:
    ///     An allocator to instantiate the block with.
    /// Returns:
    ///     An instantiated block, ready to have things added to it.
    pub fn create_block(allocator: std.mem.Allocator) Self {
        const block = std.ArrayList(*ASTNode).init(allocator);
        return Self{ .block = block };
    }

    /// Instantiates an invoke.
    /// Arguments:
    ///     invoke: Invoke - The name of the function and the arguments that go with it.
    /// Returns:
    ///     An instantiated invoke.
    pub fn create_invoke(invoke: Invoke) Self {
        return Self{ .invoke = invoke };
    }

    /// Add nodes to a block.
    /// Arguments:
    ///     node: *ASTNode - The node to add to the block.
    pub fn block_add(self: *Self, node: *ASTNode) void {
        self.block.append(node) catch unreachable;
    }

    /// Convert a node to a string representation in OpenCL C.
    /// Arguments:
    ///     allocator: std.mem.Allocator - An allocator to make the string with.
    pub fn to_opencl_string(self: *Self, allocator: std.mem.Allocator) []const u8 {
        switch (self.*) {
            .constant => |*constant| {
                // Constants are only allowed to be number values.
                switch (constant.*) {
                    .i => |*int| {
                        return std.fmt.allocPrint(allocator, "{d}", .{int.*}) catch unreachable;
                    },
                    .u => |*uint| {
                        return std.fmt.allocPrint(allocator, "{d}", .{uint.*}) catch unreachable;
                    },
                    .f => |*float| {
                        return std.fmt.allocPrint(allocator, "{d:0.15}", .{float.*}) catch unreachable;
                    },
                }
            },
            .variable => |*variable| {
                // Variables are just string names.
                return std.fmt.allocPrint(allocator, "{s}", .{variable.*}) catch unreachable;
            },
            .index => |*index| {
                // Indexes are pointer[index]
                const pointer = index.*.pointer.to_opencl_string(allocator);
                const i = index.*.index.to_opencl_string(allocator);
                defer allocator.free(pointer);
                defer allocator.free(i);
                return std.fmt.allocPrint(allocator, "{s}[{s}]", .{ pointer, i }) catch unreachable;
            },
            .assignment => |*assignment| {
                // Assignments are expression1 = expression2
                const lhs = assignment.*.lhs.to_opencl_string(allocator);
                const rhs = assignment.*.rhs.to_opencl_string(allocator);
                defer allocator.free(lhs);
                defer allocator.free(rhs);
                return std.fmt.allocPrint(allocator, "{s} = {s}", .{ lhs, rhs }) catch unreachable;
            },
            .binary_op => |*binary_op| {
                // Comparison operations are expression1 operation expression2
                const op = switch (binary_op.*.op_type) {
                    .lt => "<",
                    .lte => "<=",
                    .gt => ">",
                    .gte => ">=",
                    .eq => "==",
                    .neq => "!=",
                    .add => "+",
                    .sub => "-",
                    .mul => "*",
                    .div => "/",
                    .rem => "%",
                    .land => "&&",
                    .lor => "||",
                };
                const lhs = binary_op.*.lhs.to_opencl_string(allocator);
                const rhs = binary_op.*.rhs.to_opencl_string(allocator);
                defer allocator.free(lhs);
                defer allocator.free(rhs);
                if (binary_op.*.parentheses) {
                    return std.fmt.allocPrint(allocator, "({s} {s} {s})", .{ lhs, op, rhs }) catch unreachable;
                }
                return std.fmt.allocPrint(allocator, "{s} {s} {s}", .{ lhs, op, rhs }) catch unreachable;
            },
            .statement => |*statement| {
                // Statements are return_type expression;
                const expression = statement.*.expression.to_opencl_string(allocator);
                defer allocator.free(expression);
                if (statement.*.return_type) |return_type| {
                    switch (return_type) {
                        .int => {
                            return std.fmt.allocPrint(allocator, "int {s};", .{expression}) catch unreachable;
                        },
                        .uint => {
                            return std.fmt.allocPrint(allocator, "uint {s};", .{expression}) catch unreachable;
                        },
                        .float => {
                            return std.fmt.allocPrint(allocator, "float {s};", .{expression}) catch unreachable;
                        },
                        .int_p => {
                            return std.fmt.allocPrint(allocator, "int *{s};", .{expression}) catch unreachable;
                        },
                        .uint_p => {
                            return std.fmt.allocPrint(allocator, "uint *{s};", .{expression}) catch unreachable;
                        },
                        .float_p => {
                            return std.fmt.allocPrint(allocator, "float *{s};", .{expression}) catch unreachable;
                        },
                    }
                } else {
                    return std.fmt.allocPrint(allocator, "{s};", .{expression}) catch unreachable;
                }
            },
            .branch => |*branch| {
                // Branches are if(condition) {if_body} :else {else_body}:
                const condition = branch.*.condition.to_opencl_string(allocator);
                const if_body = branch.*.if_body.to_opencl_string(allocator);
                defer allocator.free(condition);
                defer allocator.free(if_body);
                if (branch.*.else_body) |else_body_node| {
                    const else_body = else_body_node.*.to_opencl_string(allocator);
                    defer allocator.free(else_body);
                    return std.fmt.allocPrint(allocator, "if({s}) {{{s}}} else {{{s}}}", .{ condition, if_body, else_body }) catch unreachable;
                }
                return std.fmt.allocPrint(allocator, "if({s}) {{{s}}}", .{ condition, if_body }) catch unreachable;
            },
            .while_loop => |*while_loop| {
                // While loops are while(condition) {body}
                const condition = while_loop.*.condition.to_opencl_string(allocator);
                const body = while_loop.*.body.to_opencl_string(allocator);
                defer allocator.free(condition);
                defer allocator.free(body);
                return std.fmt.allocPrint(allocator, "while({s}) {{{s}}}", .{ condition, body }) catch unreachable;
            },
            .block => |*block| {
                // Blocks are node1 node2 node3...
                var buf = std.ArrayList(u8).init(allocator);
                defer buf.deinit();

                for (block.*.items, 0..) |child, i| {
                    const str_val = child.*.to_opencl_string(allocator);
                    defer allocator.free(str_val);
                    buf.appendSlice(str_val) catch unreachable;
                    if (i < block.*.items.len - 1) {
                        buf.append(' ') catch unreachable;
                    }
                }

                return buf.toOwnedSlice() catch unreachable;
            },
            .invoke => |*invoke| {
                // Invokes are funciton_name(argument1, argument2, ...)
                var buf = std.ArrayList(u8).init(allocator);
                defer buf.deinit();

                for (invoke.*.args, 0..) |arg, i| {
                    const str_val = arg.*.to_opencl_string(allocator);
                    defer allocator.free(str_val);
                    buf.appendSlice(str_val) catch unreachable;
                    if (i < invoke.*.args.len - 1) {
                        buf.appendSlice(", ") catch unreachable;
                    }
                }

                const arg_str = buf.toOwnedSlice() catch unreachable;

                return std.fmt.allocPrint(allocator, "{s}({s})", .{ invoke.*.name, arg_str }) catch unreachable;
            },
        }
    }
};

pub const Kernel = struct {
    /// Allows for using ASTNode in a linked list.
    const CtxNode = struct {
        data: *ASTNode,
        node: std.SinglyLinkedList.Node = .{},
    };

    /// Allows for easier building of expressions.
    const ExpressionBuilder = struct {
        /// ast: *AST - A pointer to an abstract syntax tree, used to grab variable nodes.
        ast: *Kernel,

        const xbSelf = @This();

        /// Return an initialized expression builder.
        /// Arguments:
        ///     ast: *AST - Pointer to parent AST.
        pub fn init(ast: *Kernel) xbSelf {
            return xbSelf{ .ast = ast };
        }

        /// Create a constant node.
        /// Arguments:
        ///     comptime T: type - The type of the constant.
        ///     value: T - The value of the constant.
        /// Returns:
        ///     A constant *ASTNode.
        pub fn constant(self: *xbSelf, comptime T: type, value: T) *ASTNode {
            const const_node = self.ast.allocator.create(ASTNode) catch unreachable;
            const_node.* = ASTNode.create_constant(T, value);
            return const_node;
        }

        /// Lookup and return the variable node associated with name in the parent AST.
        /// Arguments:
        ///     name: []const u8 - The name associated with the variable.
        /// Returns:
        ///     A variable *ASTNode.
        pub fn vars(self: *xbSelf, name: []const u8) *ASTNode {
            const has_var = self.ast.identifiers.get(name);
            if (has_var) |var_node| {
                return var_node;
            }
            unreachable;
        }

        /// Add a series of nodes. Surround them with parentheses.
        /// Arguments:
        ///     nodes: A list of nodes to add togehter.
        /// Returns:
        ///     A binary operation composed of smaller operations (all nodes included).
        pub fn add(self: *xbSelf, nodes: []const *ASTNode) *ASTNode {

            // Get the number of nodes in the list
            const len = nodes.len;

            // Make sure the list is at least two nodes
            std.debug.assert(len >= 2);

            var result = self.ast.allocator.create(ASTNode) catch unreachable;

            // If length equals 2, we need parentheses
            if (len == 2) {
                result.* = ASTNode.create_binary_op(.{ .lhs = nodes[0], .rhs = nodes[1], .op_type = .add, .parentheses = true });
                return result;
            }

            result.* = ASTNode.create_binary_op(.{ .lhs = nodes[0], .rhs = nodes[1], .op_type = .add, .parentheses = false });

            // Recursively build up the expression
            var i: usize = 2;
            while (i < len) : (i += 1) {
                const temp = result;
                var parentheses = true;
                if (i < len - 1) {
                    parentheses = false;
                }
                result = self.ast.allocator.create(ASTNode) catch unreachable;
                result.* = ASTNode.create_binary_op(.{ .lhs = temp, .rhs = nodes[i], .op_type = .add, .parentheses = parentheses });
            }

            return result;
        }

        /// Subtract a series of nodes. Surround them with parentheses.
        /// Arguments:
        ///     nodes: A list of nodes to subtract togehter.
        /// Returns:
        ///     A binary operation composed of smaller operations (all nodes included).
        pub fn sub(self: *xbSelf, nodes: []const *ASTNode) *ASTNode {
            // Get the number of nodes in the list
            const len = nodes.len;

            // Make sure the list is at least two nodes
            std.debug.assert(len >= 2);

            var result = self.ast.allocator.create(ASTNode) catch unreachable;

            // If length equals 2, we need parentheses
            if (len == 2) {
                result.* = ASTNode.create_binary_op(.{ .lhs = nodes[0], .rhs = nodes[1], .op_type = .sub, .parentheses = true });
                return result;
            }

            result.* = ASTNode.create_binary_op(.{ .lhs = nodes[0], .rhs = nodes[1], .op_type = .sub, .parentheses = false });

            // Recursively build up the expression
            var i: usize = 2;
            while (i < len) : (i += 1) {
                const temp = result;
                var parentheses = true;
                if (i < len - 1) {
                    parentheses = false;
                }
                result = self.ast.allocator.create(ASTNode) catch unreachable;
                result.* = ASTNode.create_binary_op(.{ .lhs = temp, .rhs = nodes[i], .op_type = .sub, .parentheses = parentheses });
            }

            return result;
        }

        /// Multiply a series of nodes. Surround them with parentheses.
        /// Arguments:
        ///     nodes: A list of nodes to multiply togehter.
        /// Returns:
        ///     A binary operation composed of smaller operations (all nodes included).
        pub fn mul(self: *xbSelf, nodes: []const *ASTNode) *ASTNode {
            // Get the number of nodes in the list
            const len = nodes.len;

            // Make sure the list is at least two nodes
            std.debug.assert(len >= 2);

            var result = self.ast.allocator.create(ASTNode) catch unreachable;

            // If length equals 2, we need parentheses
            if (len == 2) {
                result.* = ASTNode.create_binary_op(.{ .lhs = nodes[0], .rhs = nodes[1], .op_type = .mul, .parentheses = true });
                return result;
            }

            result.* = ASTNode.create_binary_op(.{ .lhs = nodes[0], .rhs = nodes[1], .op_type = .mul, .parentheses = false });

            // Recursively build up the expression
            var i: usize = 2;
            while (i < len) : (i += 1) {
                const temp = result;
                var parentheses = true;
                if (i < len - 1) {
                    parentheses = false;
                }
                result = self.ast.allocator.create(ASTNode) catch unreachable;
                result.* = ASTNode.create_binary_op(.{ .lhs = temp, .rhs = nodes[i], .op_type = .mul, .parentheses = parentheses });
            }

            return result;
        }

        /// Divide a series of nodes. Surround them with parentheses.
        /// Arguments:
        ///     nodes: A list of nodes to divide togehter.
        /// Returns:
        ///     A binary operation composed of smaller operations (all nodes included).
        pub fn div(self: *xbSelf, nodes: []const *ASTNode) *ASTNode {
            // Get the number of nodes in the list
            const len = nodes.len;

            // Make sure the list is at least two nodes
            std.debug.assert(len >= 2);

            var result = self.ast.allocator.create(ASTNode) catch unreachable;

            // If length equals 2, we need parentheses
            if (len == 2) {
                result.* = ASTNode.create_binary_op(.{ .lhs = nodes[0], .rhs = nodes[1], .op_type = .div, .parentheses = true });
                return result;
            }

            result.* = ASTNode.create_binary_op(.{ .lhs = nodes[0], .rhs = nodes[1], .op_type = .div, .parentheses = false });

            // Recursively build up the expression
            var i: usize = 2;
            while (i < len) : (i += 1) {
                const temp = result;
                var parentheses = true;
                if (i < len - 1) {
                    parentheses = false;
                }
                result = self.ast.allocator.create(ASTNode) catch unreachable;
                result.* = ASTNode.create_binary_op(.{ .lhs = temp, .rhs = nodes[i], .op_type = .div, .parentheses = parentheses });
            }

            return result;
        }

        /// Modulus a series of nodes. Surround them with parentheses.
        /// Arguments:
        ///     nodes: A list of nodes to mod togehter.
        /// Returns:
        ///     A binary operation composed of smaller operations (all nodes included).
        pub fn mod(self: *xbSelf, nodes: []const *ASTNode) *ASTNode {
            // Get the number of nodes in the list
            const len = nodes.len;

            // Make sure the list is at least two nodes
            std.debug.assert(len >= 2);

            var result = self.ast.allocator.create(ASTNode) catch unreachable;

            // If length equals 2, we need parentheses
            if (len == 2) {
                result.* = ASTNode.create_binary_op(.{ .lhs = nodes[0], .rhs = nodes[1], .op_type = .rem, .parentheses = true });
                return result;
            }

            result.* = ASTNode.create_binary_op(.{ .lhs = nodes[0], .rhs = nodes[1], .op_type = .rem, .parentheses = false });

            // Recursively build up the expression
            var i: usize = 2;
            while (i < len) : (i += 1) {
                const temp = result;
                var parentheses = true;
                if (i < len - 1) {
                    parentheses = false;
                }
                result = self.ast.allocator.create(ASTNode) catch unreachable;
                result.* = ASTNode.create_binary_op(.{ .lhs = temp, .rhs = nodes[i], .op_type = .rem, .parentheses = parentheses });
            }

            return result;
        }

        /// Greater than comparison of two nodes.
        /// Arguments:
        ///     lhs: *ASTNode - The left hand side of the expression.
        ///     rhs: *ASTNode - The right hand side of the expression.
        /// Returns:
        ///     A binary operation composed of the left hand side and right hand side.
        pub fn gt(self: *xbSelf, lhs: *ASTNode, rhs: *ASTNode) *ASTNode {
            const result = self.ast.allocator.create(ASTNode) catch unreachable;
            result.* = ASTNode.create_binary_op(.{ .lhs = lhs, .rhs = rhs, .op_type = .gt, .parentheses = false });
            return result;
        }

        /// Greater than or equal comparison of two nodes.
        /// Arguments:
        ///     lhs: *ASTNode - The left hand side of the expression.
        ///     rhs: *ASTNode - The right hand side of the expression.
        /// Returns:
        ///     A binary operation composed of the left hand side and right hand side.
        pub fn gte(self: *xbSelf, lhs: *ASTNode, rhs: *ASTNode) *ASTNode {
            const result = self.ast.allocator.create(ASTNode) catch unreachable;
            result.* = ASTNode.create_binary_op(.{ .lhs = lhs, .rhs = rhs, .op_type = .gte, .parentheses = false });
            return result;
        }

        /// Lesser than comparison of two nodes.
        /// Arguments:
        ///     lhs: *ASTNode - The left hand side of the expression.
        ///     rhs: *ASTNode - The right hand side of the expression.
        /// Returns:
        ///     A binary operation composed of the left hand side and right hand side.
        pub fn lt(self: *xbSelf, lhs: *ASTNode, rhs: *ASTNode) *ASTNode {
            const result = self.ast.allocator.create(ASTNode) catch unreachable;
            result.* = ASTNode.create_binary_op(.{ .lhs = lhs, .rhs = rhs, .op_type = .lt, .parentheses = false });
            return result;
        }

        /// Lesser than or equal comparison of two nodes.
        /// Arguments:
        ///     lhs: *ASTNode - The left hand side of the expression.
        ///     rhs: *ASTNode - The right hand side of the expression.
        /// Returns:
        ///     A binary operation composed of the left hand side and right hand side.
        pub fn lte(self: *xbSelf, lhs: *ASTNode, rhs: *ASTNode) *ASTNode {
            const result = self.ast.allocator.create(ASTNode) catch unreachable;
            result.* = ASTNode.create_binary_op(.{ .lhs = lhs, .rhs = rhs, .op_type = .lte, .parentheses = false });
            return result;
        }

        /// Equality comparison of two nodes.
        /// Arguments:
        ///     lhs: *ASTNode - The left hand side of the expression.
        ///     rhs: *ASTNode - The right hand side of the expression.
        /// Returns:
        ///     A binary operation composed of the left hand side and right hand side.
        pub fn eq(self: *xbSelf, lhs: *ASTNode, rhs: *ASTNode) *ASTNode {
            const result = self.ast.allocator.create(ASTNode) catch unreachable;
            result.* = ASTNode.create_binary_op(.{ .lhs = lhs, .rhs = rhs, .op_type = .eq, .parentheses = false });
            return result;
        }

        /// Non-equality comparison of two nodes.
        /// Arguments:
        ///     lhs: *ASTNode - The left hand side of the expression.
        ///     rhs: *ASTNode - The right hand side of the expression.
        /// Returns:
        ///     A binary operation composed of the left hand side and right hand side.
        pub fn neq(self: *xbSelf, lhs: *ASTNode, rhs: *ASTNode) *ASTNode {
            const result = self.ast.allocator.create(ASTNode) catch unreachable;
            result.* = ASTNode.create_binary_op(.{ .lhs = lhs, .rhs = rhs, .op_type = .neq, .parentheses = false });
            return result;
        }

        /// Logical AND a series of nodes. Surround them with parentheses.
        /// Arguments:
        ///     nodes: A list of nodes to AND togehter.
        /// Returns:
        ///     A binary operation composed of smaller operations (all nodes included).
        pub fn land(self: *xbSelf, nodes: []const *ASTNode) *ASTNode {
            // Get the number of nodes in the list
            const len = nodes.len;

            // Make sure the list is at least two nodes
            std.debug.assert(len >= 2);

            var result = self.ast.allocator.create(ASTNode) catch unreachable;

            // If length equals 2, we need parentheses
            if (len == 2) {
                result.* = ASTNode.create_binary_op(.{ .lhs = nodes[0], .rhs = nodes[1], .op_type = .land, .parentheses = true });
                return result;
            }

            result.* = ASTNode.create_binary_op(.{ .lhs = nodes[0], .rhs = nodes[1], .op_type = .land, .parentheses = false });

            // Recursively build up the expression
            var i: usize = 2;
            while (i < len) : (i += 1) {
                const temp = result;
                var parentheses = true;
                if (i < len - 1) {
                    parentheses = false;
                }
                result = self.ast.allocator.create(ASTNode) catch unreachable;
                result.* = ASTNode.create_binary_op(.{ .lhs = temp, .rhs = nodes[i], .op_type = .land, .parentheses = parentheses });
            }

            return result;
        }

        /// Logical OR a series of nodes. Surround them with parentheses.
        /// Arguments:
        ///     nodes: A list of nodes to OR togehter.
        /// Returns:
        ///     A binary operation composed of smaller operations (all nodes included).
        pub fn lor(self: *xbSelf, nodes: []const *ASTNode) *ASTNode {
            // Get the number of nodes in the list
            const len = nodes.len;

            // Make sure the list is at least two nodes
            std.debug.assert(len >= 2);

            var result = self.ast.allocator.create(ASTNode) catch unreachable;

            // If length equals 2, we need parentheses
            if (len == 2) {
                result.* = ASTNode.create_binary_op(.{ .lhs = nodes[0], .rhs = nodes[1], .op_type = .lor, .parentheses = true });
                return result;
            }

            result.* = ASTNode.create_binary_op(.{ .lhs = nodes[0], .rhs = nodes[1], .op_type = .lor, .parentheses = false });

            // Recursively build up the expression
            var i: usize = 2;
            while (i < len) : (i += 1) {
                const temp = result;
                var parentheses = true;
                if (i < len - 1) {
                    parentheses = false;
                }
                result = self.ast.allocator.create(ASTNode) catch unreachable;
                result.* = ASTNode.create_binary_op(.{ .lhs = temp, .rhs = nodes[i], .op_type = .lor, .parentheses = parentheses });
            }

            return result;
        }

        /// Index into a pointer type.
        /// Arguments:
        ///     pointer: *ASTNode - The node to index into.
        ///     i: *ASTNode - The index value.
        /// Returns:
        ///     An index composed of the pointer and index provided.
        pub fn index(self: *xbSelf, pointer: *ASTNode, i: *ASTNode) *ASTNode {
            const result = self.ast.allocator.create(ASTNode) catch unreachable;

            result.* = ASTNode.create_index(.{ .pointer = pointer, .index = i });

            return result;
        }

        /// Invoke a function.
        /// Arguments:
        ///     name: []const u8 - The name of the function.
        ///     args: []const *ASTNode - A list of arguments to pass into the function.
        /// Returns:
        ///     An invoke composed of the name and arguments provided.
        pub fn invoke(self: *xbSelf, name: []const u8, args: []const *ASTNode) *ASTNode {
            const result = self.ast.allocator.create(ASTNode) catch unreachable;

            result.* = ASTNode.create_invoke(.{ .name = name, .args = args });

            return result;
        }
    };

    /// root: *ASTNode - The root block node of the AST.
    /// kernel_name: []const u8 - A name to assign the kernel.
    /// arg_types: []const type - The types of the arguments passed to the kernel.
    /// args: [][]const u8 - The arguments passed to the kernel.
    /// identifiers: std.StringHashMap(*ASTNode) - Maps variable names to the nodes that represent them.
    /// context: std.SinglyLinkedList - Keeps track of where you are in the AST (so you can add several statements to the same while loop for example).
    /// allocator: std.mem.Allocator - The arena's allocator.
    root: *ASTNode,
    kernel_name: []const u8,
    args: []const []const u8,
    identifiers: std.StringHashMap(*ASTNode),
    context: std.SinglyLinkedList,
    allocator: std.mem.Allocator,

    const Self = @This();

    /// Initialize an Abstract Syntax Tree.
    /// Arguments:
    ///     name: []const u8 - A name to give the kernel.
    ///     arg_type: []const type - The types of the arguments.
    ///     args: [][]const u8 - A list of arguments to pass the kernel.
    ///     allocator: std.mem.Allocator - An allocator to manage memory (make sure this is an arena allocator).
    /// Returns:
    ///     An instantiated AST object.
    pub fn init(name: []const u8, args: []const []const u8, allocator: std.mem.Allocator) !Self {
        // Create the top level block statement
        const root = try allocator.create(ASTNode);
        root.* = ASTNode.create_block(allocator);

        var identifiers = std.StringHashMap(*ASTNode).init(allocator);

        for (args) |arg| {
            const var_node = allocator.create(ASTNode) catch unreachable;
            var_node.* = ASTNode.create_variable(arg);
            identifiers.put(arg, var_node) catch unreachable;
        }

        var list: std.SinglyLinkedList = .{};

        // Allocate the head CtxNode on the heap
        const top_level_block = try allocator.create(CtxNode);
        top_level_block.* = .{ .data = root };
        list.prepend(&top_level_block.node);

        return Self{ .root = root, .kernel_name = name, .args = args, .identifiers = identifiers, .context = list, .allocator = allocator };
    }

    /// Destroy an AST object.
    pub fn deinit(self: *Self) void {
        self._end_block();
        self.identifiers.deinit();
        self.root.deinit(self.allocator);
    }

    /// Create an expression builder for this kernel.
    /// Returns:
    ///     An ExpressionBuilder with the parent AST set to this kernel.
    pub fn expression_builder(self: *Self) ExpressionBuilder {
        return ExpressionBuilder.init(self);
    }

    /// Declare a variable in the current block. Do not initialize.
    /// Arguments:
    ///     comptime T: type - The type of variable to initialize.
    ///     name: []const u8 - A name to give the variable.
    pub fn _declare(self: *Self, comptime T: type, name: []const u8) void {
        switch (@typeInfo(T)) {
            .int => |i| {
                if (self.context.first) |head| {
                    // Create a variable node
                    const var_node = self.allocator.create(ASTNode) catch unreachable;
                    var_node.* = ASTNode.create_variable(name);

                    // Create the declaration statement
                    const statement_node = self.allocator.create(ASTNode) catch unreachable;
                    if (i.signedness == .signed) {
                        statement_node.* = ASTNode.create_statement(.{ .expression = var_node, .return_type = .int });
                    } else {
                        statement_node.* = ASTNode.create_statement(.{ .expression = var_node, .return_type = .uint });
                    }

                    // Grab the current block that this declaration should go in.
                    const node: *CtxNode = @fieldParentPtr("node", head);
                    var current_block = node.data;

                    // Add the statement to the current block
                    current_block.block_add(statement_node);

                    // Add the variable to the identifiers lookup
                    self.identifiers.put(name, var_node) catch unreachable;
                }
            },
            .float => {
                if (self.context.first) |head| {
                    // Create a variable node
                    const var_node = self.allocator.create(ASTNode) catch unreachable;
                    var_node.* = ASTNode.create_variable(name);

                    // Create the declaration statement
                    const statement_node = self.allocator.create(ASTNode) catch unreachable;
                    statement_node.* = ASTNode.create_statement(.{ .expression = var_node, .return_type = .float });

                    // Grab the current block that this declaration should go in.
                    const node: *CtxNode = @fieldParentPtr("node", head);
                    var current_block = node.data;

                    // Add the statement to the current block
                    current_block.block_add(statement_node);

                    // Add the variable to the identifiers lookup
                    self.identifiers.put(name, var_node) catch unreachable;
                }
            },
            .pointer => |ptr| {
                switch (@typeInfo(ptr.child)) {
                    .int => |i| {
                        if (self.context.first) |head| {
                            // Create a variable node
                            const var_node = self.allocator.create(ASTNode) catch unreachable;
                            var_node.* = ASTNode.create_variable(name);

                            // Create the declaration statement
                            const statement_node = self.allocator.create(ASTNode) catch unreachable;
                            if (i.signedness == .signed) {
                                statement_node.* = ASTNode.create_statement(.{ .expression = var_node, .return_type = .int_p });
                            } else {
                                statement_node.* = ASTNode.create_statement(.{ .expression = var_node, .return_type = .uint_p });
                            }

                            // Grab the current block that this declaration should go in.
                            const node: *CtxNode = @fieldParentPtr("node", head);
                            var current_block = node.data;

                            // Add the statement to the current block
                            current_block.block_add(statement_node);

                            // Add the variable to the identifiers lookup
                            self.identifiers.put(name, var_node) catch unreachable;
                        }
                    },
                    .float => {
                        if (self.context.first) |head| {
                            // Create a variable node
                            const var_node = self.allocator.create(ASTNode) catch unreachable;
                            var_node.* = ASTNode.create_variable(name);

                            // Create the declaration statement
                            const statement_node = self.allocator.create(ASTNode) catch unreachable;
                            statement_node.* = ASTNode.create_statement(.{ .expression = var_node, .return_type = .float });

                            // Grab the current block that this declaration should go in.
                            const node: *CtxNode = @fieldParentPtr("node", head);
                            var current_block = node.data;

                            // Add the statement to the current block
                            current_block.block_add(statement_node);

                            // Add the variable to the identifiers lookup
                            self.identifiers.put(name, var_node) catch unreachable;
                        }
                    },
                    else => unreachable,
                }
            },
            else => unreachable,
        }
    }

    /// Assign a rhs to a lhs in the current block.
    /// Arguments:
    ///     lhs: *ASTNode - The left hand side of the statement.
    ///     rhs: *ASTNode - The right hand side of the statement.
    pub fn _assign(self: *Self, lhs: *ASTNode, rhs: *ASTNode) void {
        if (self.context.first) |head| {
            // Create the assignment node
            const assign_node = self.allocator.create(ASTNode) catch unreachable;
            assign_node.* = ASTNode.create_assignment(.{ .lhs = lhs, .rhs = rhs });

            // Create the statement node
            const statement_node = self.allocator.create(ASTNode) catch unreachable;
            statement_node.* = ASTNode.create_statement(.{ .expression = assign_node, .return_type = null });

            // Grab the current block that this assignment should go in
            const node: *CtxNode = @fieldParentPtr("node", head);
            var current_block = node.data;

            current_block.block_add(statement_node);
        }
    }

    /// Declare and initialize a variable. Set the value of the variable to a statement.
    /// Arguments:
    ///     comptime T: type - The type to assign the variable.
    ///     name: []const u8 - A name to give to the variable.
    ///     expression: *ASTNode - A statement to set the variable to.
    pub fn _initialize(self: *Self, comptime T: type, name: []const u8, expression: *ASTNode) void {
        switch (@typeInfo(T)) {
            .int => |i| {
                if (self.context.first) |head| {
                    // Create a variable node
                    const var_node = self.allocator.create(ASTNode) catch unreachable;
                    var_node.* = ASTNode.create_variable(name);

                    // Create the assignment node
                    const assign_node = self.allocator.create(ASTNode) catch unreachable;
                    assign_node.* = ASTNode.create_assignment(.{ .lhs = var_node, .rhs = expression });

                    // Create the initialization statement
                    const statement_node = self.allocator.create(ASTNode) catch unreachable;
                    if (i.signedness == .signed) {
                        statement_node.* = ASTNode.create_statement(.{ .expression = assign_node, .return_type = .int });
                    } else {
                        statement_node.* = ASTNode.create_statement(.{ .expression = assign_node, .return_type = .uint });
                    }

                    // Grab the current block that this declaration should go in.
                    const node: *CtxNode = @fieldParentPtr("node", head);
                    var current_block = node.data;

                    // Add the statement to the current block
                    current_block.block_add(statement_node);

                    // Add the variable to the identifiers lookup
                    self.identifiers.put(name, var_node) catch unreachable;
                }
            },
            .float => {
                if (self.context.first) |head| {
                    // Create a variable node
                    const var_node = self.allocator.create(ASTNode) catch unreachable;
                    var_node.* = ASTNode.create_variable(name);

                    // Create the assignment node
                    const assign_node = self.allocator.create(ASTNode) catch unreachable;
                    assign_node.* = ASTNode.create_assignment(.{ .lhs = var_node, .rhs = expression });

                    // Create the initialization statement
                    const statement_node = self.allocator.create(ASTNode) catch unreachable;
                    statement_node.* = ASTNode.create_statement(.{ .expression = assign_node, .return_type = .float });

                    // Grab the current block that this declaration should go in.
                    const node: *CtxNode = @fieldParentPtr("node", head);
                    var current_block = node.data;

                    // Add the statement to the current block
                    current_block.block_add(statement_node);

                    // Add the variable to the identifiers lookup
                    self.identifiers.put(name, var_node) catch unreachable;
                }
            },
            .pointer => |ptr| {
                switch (@typeInfo(ptr.child)) {
                    .int => |i| {
                        if (self.context.first) |head| {
                            // Create a variable node
                            const var_node = self.allocator.create(ASTNode) catch unreachable;
                            var_node.* = ASTNode.create_variable(name);

                            // Create the assignment node
                            const assign_node = self.allocator.create(ASTNode) catch unreachable;
                            assign_node.* = ASTNode.create_assignment(.{ .lhs = var_node, .rhs = expression });

                            // Create the declaration statement
                            const statement_node = self.allocator.create(ASTNode) catch unreachable;
                            if (i.signedness == .signed) {
                                statement_node.* = ASTNode.create_statement(.{ .expression = assign_node, .return_type = .int_p });
                            } else {
                                statement_node.* = ASTNode.create_statement(.{ .expression = assign_node, .return_type = .uint_p });
                            }

                            // Grab the current block that this declaration should go in.
                            const node: *CtxNode = @fieldParentPtr("node", head);
                            var current_block = node.data;

                            // Add the statement to the current block
                            current_block.block_add(statement_node);

                            // Add the variable to the identifiers lookup
                            self.identifiers.put(name, var_node) catch unreachable;
                        }
                    },
                    .float => {
                        if (self.context.first) |head| {
                            // Create a variable node
                            const var_node = self.allocator.create(ASTNode) catch unreachable;
                            var_node.* = ASTNode.create_variable(name);

                            // Create the assignment node
                            const assign_node = self.allocator.create(ASTNode) catch unreachable;
                            assign_node.* = ASTNode.create_assignment(.{ .lhs = var_node, .rhs = expression });

                            // Create the declaration statement
                            const statement_node = self.allocator.create(ASTNode) catch unreachable;
                            statement_node.* = ASTNode.create_statement(.{ .expression = assign_node, .return_type = .float });

                            // Grab the current block that this declaration should go in.
                            const node: *CtxNode = @fieldParentPtr("node", head);
                            var current_block = node.data;

                            // Add the statement to the current block
                            current_block.block_add(statement_node);

                            // Add the variable to the identifiers lookup
                            self.identifiers.put(name, var_node) catch unreachable;
                        }
                    },
                    else => unreachable,
                }
            },
            else => unreachable,
        }
    }

    /// Create and enter a while loop.
    /// Arguments:
    ///     condition: *ASTNode - The condition to terminate the while loop.
    pub fn _while(self: *Self, condition: *ASTNode) void {
        if (self.context.first) |head| {
            // Create a block node for the body
            const body_node = self.allocator.create(ASTNode) catch unreachable;
            body_node.* = ASTNode.create_block(self.allocator);

            // Create a while node
            const while_node = self.allocator.create(ASTNode) catch unreachable;
            while_node.* = ASTNode.create_while_loop(.{ .condition = condition, .body = body_node });

            // Grab the current block that this declaration should go in
            const node: *CtxNode = @fieldParentPtr("node", head);
            var current_block = node.data;

            // Add the while loop to the current block
            current_block.block_add(while_node);

            // The new current block is now the while loop body
            const new_context = self.allocator.create(CtxNode) catch unreachable;
            new_context.* = .{ .data = body_node };
            self.context.prepend(&new_context.node);
        }
    }

    /// Create and enter an if block.
    /// Arguments:
    ///     condition: *ASTNode - The condition to enter the if statement.
    pub fn _if(self: *Self, condition: *ASTNode) void {
        if (self.context.first) |head| {
            // Create a block node for the if body
            const if_body = self.allocator.create(ASTNode) catch unreachable;
            if_body.* = ASTNode.create_block(self.allocator);

            // Create a branch node
            const if_node = self.allocator.create(ASTNode) catch unreachable;
            if_node.* = ASTNode.create_branch(.{ .condition = condition, .if_body = if_body });

            // Grab the current block that this declaration should go in
            const node: *CtxNode = @fieldParentPtr("node", head);
            var current_block = node.data;

            // Add the if statement to the current block
            current_block.block_add(if_node);

            // The new current block is now the if body
            const new_context = self.allocator.create(CtxNode) catch unreachable;
            new_context.* = .{ .data = if_body };
            self.context.prepend(&new_context.node);
        }
    }

    /// Create and enter an else block.
    pub fn _else(self: *Self) void {
        // Need to remove the current if statement
        if (self.context.popFirst()) |head| {
            // Grab the current block
            const node: *CtxNode = @fieldParentPtr("node", head);

            // Destroy the current context node
            self.allocator.destroy(node);

            // Grab the ASTNode associated with the block
            if (self.context.first) |new_head| {
                // Grab the new current block
                const new_node: *CtxNode = @fieldParentPtr("node", new_head);
                const new_current_block = new_node.data;

                // The branch node we are looking for should be the last node in the new current context
                const ast_node = new_current_block.block.items[new_current_block.block.items.len - 1];

                switch (ast_node.*) {
                    .branch => |*branch| {
                        // Create a block node for the else body
                        const else_body = self.allocator.create(ASTNode) catch unreachable;
                        else_body.* = ASTNode.create_block(self.allocator);

                        // Set the branch's else body
                        branch.else_body = else_body;

                        // The new current block is now the else body
                        const new_context = self.allocator.create(CtxNode) catch unreachable;
                        new_context.* = .{ .data = else_body };
                        self.context.prepend(&new_context.node);
                    },
                    else => unreachable,
                }
            }
        }
    }

    /// End a block (i.e., jump out of the block's body)
    pub fn _end_block(self: *Self) void {
        if (self.context.popFirst()) |head| {
            const node: *CtxNode = @fieldParentPtr("node", head);
            self.allocator.destroy(node);
        }
    }

    /// Call a function and return nothing.
    /// Arguments:
    ///     invoke: *ASTNode - The invoke to wrap this statement around
    pub fn _call(self: *Self, invoke: *ASTNode) void {
        if (self.context.first) |head| {
            // Wrap the invoke in a statement
            const statement_node = self.allocator.create(ASTNode) catch unreachable;
            statement_node.* = ASTNode.create_statement(.{ .expression = invoke });

            // Get the current block to add this call to
            const node: *CtxNode = @fieldParentPtr("node", head);
            var current_block = node.data;

            current_block.block_add(statement_node);
        }
    }

    /// Convert the kernel to an OpenCL C string.
    /// Arguments:
    ///     comptime argtypes: []const type - The types of the arguments specified in the initialization.
    /// Returns:
    ///     A string representation of the kernel in OpenCL C.
    pub fn to_opencl_string(self: *Self, comptime arg_types: []const type) []const u8 {
        // Make sure the types match the arguments
        std.debug.assert(arg_types.len == self.args.len);

        // Determine the body of the kernel up front.
        const block_str = self.root.to_opencl_string(self.allocator);
        defer self.allocator.free(block_str);

        // A buffer to build the wrapped kernel around the body
        var buf = std.ArrayList(u8).init(self.allocator);
        defer buf.deinit();

        // Kernels are defined as __kernel void kernel_name(__global arg1, __global arg2, ...) {body}
        buf.appendSlice("__kernel void ") catch unreachable;
        buf.appendSlice(self.kernel_name) catch unreachable;
        buf.append('(') catch unreachable;

        // Make this inline since the types are comptime
        inline for (0..arg_types.len) |i| {
            buf.appendSlice("__global ") catch unreachable;
            switch (@typeInfo(arg_types[i])) {
                .int => |int| {
                    if (int.signedness == .signed) {
                        buf.appendSlice("int ") catch unreachable;
                    } else {
                        buf.appendSlice("uint ") catch unreachable;
                    }
                },
                .float => {
                    buf.appendSlice("float ") catch unreachable;
                },
                else => unreachable,
                .pointer => |ptr_info| {
                    switch (@typeInfo(ptr_info.child)) {
                        .int => |int| {
                            if (int.signedness == .signed) {
                                buf.appendSlice("int *") catch unreachable;
                            } else {
                                buf.appendSlice("uint *") catch unreachable;
                            }
                        },
                        .float => {
                            buf.appendSlice("float *") catch unreachable;
                        },
                        else => unreachable,
                    }
                },
            }
            buf.appendSlice(self.args[i]) catch unreachable;
            if (i < self.args.len - 1) {
                buf.appendSlice(", ") catch unreachable;
            }
        }
        buf.appendSlice(") {") catch unreachable;

        // We are now in the body
        buf.appendSlice(block_str) catch unreachable;

        // Finish the kernel with a closing curly brace
        buf.append('}') catch unreachable;

        return buf.toOwnedSlice() catch unreachable;
    }
};

test "ASTNode test" {
    // Test the abilities of the ASTNodes individually
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    const node1 = try allocator.create(ASTNode);
    const node2 = try allocator.create(ASTNode);
    const node3 = try allocator.create(ASTNode);
    const node4 = try allocator.create(ASTNode);
    const node5 = try allocator.create(ASTNode);
    const node6 = try allocator.create(ASTNode);
    const node7 = try allocator.create(ASTNode);
    const node8 = try allocator.create(ASTNode);
    const node9 = try allocator.create(ASTNode);
    const node10 = try allocator.create(ASTNode);
    const node11 = try allocator.create(ASTNode);
    const node12 = try allocator.create(ASTNode);
    const node13 = try allocator.create(ASTNode);
    const node14 = try allocator.create(ASTNode);
    const node15 = try allocator.create(ASTNode);
    const node16 = try allocator.create(ASTNode);
    const node17 = try allocator.create(ASTNode);
    const node18 = try allocator.create(ASTNode);
    const node19 = try allocator.create(ASTNode);
    const node20 = try allocator.create(ASTNode);

    node1.* = ASTNode.create_constant(usize, 0);
    node2.* = ASTNode.create_constant(usize, 100);
    node3.* = ASTNode.create_constant(usize, 50);

    node4.* = ASTNode.create_variable("a");
    node5.* = ASTNode.create_variable("b");

    node6.* = ASTNode.create_assignment(.{ .lhs = node4, .rhs = node2 });
    node7.* = ASTNode.create_statement(.{ .return_type = .uint, .expression = node6 });

    node8.* = ASTNode.create_assignment(.{ .lhs = node5, .rhs = node3 });
    node9.* = ASTNode.create_statement(.{ .return_type = .uint, .expression = node8 });

    node10.* = ASTNode.create_binary_op(.{ .op_type = .neq, .lhs = node5, .rhs = node1 });
    node11.* = ASTNode.create_binary_op(.{ .op_type = .gt, .lhs = node4, .rhs = node5 });
    node12.* = ASTNode.create_binary_op(.{ .op_type = .sub, .lhs = node4, .rhs = node5 });
    node13.* = ASTNode.create_binary_op(.{ .op_type = .sub, .lhs = node5, .rhs = node4 });

    node14.* = ASTNode.create_assignment(.{ .lhs = node4, .rhs = node12 });
    node15.* = ASTNode.create_assignment(.{ .lhs = node5, .rhs = node13 });

    node16.* = ASTNode.create_statement(.{ .expression = node14 });
    node17.* = ASTNode.create_statement(.{ .expression = node15 });

    node18.* = ASTNode.create_branch(.{ .condition = node11, .if_body = node16, .else_body = node17 });

    node19.* = ASTNode.create_while_loop(.{ .condition = node10, .body = node18 });

    node20.* = ASTNode.create_block(allocator);
    node20.block_add(node7);
    node20.block_add(node9);
    node20.block_add(node19);

    const program = node20.to_opencl_string(allocator);
    defer allocator.free(program);
    const expected_program = "uint a = 100; uint b = 50; while(b != 0) {if(a > b) {a = a - b;} else {b = b - a;}}";
    try std.testing.expect(std.mem.eql(u8, program, expected_program));
}

test "AST test" {
    // Test the abilitiy of the Kernel struct for building kernels
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    var kernel = try Kernel.init("vadd", &.{ "a", "b", "c" }, allocator);
    defer kernel.deinit();

    var xb = kernel.expression_builder();

    kernel._initialize(usize, "id", xb.invoke("get_global_id", &.{xb.constant(usize, 0)}));
    kernel._assign(xb.index(xb.vars("c"), xb.vars("id")), xb.add(&.{ xb.index(xb.vars("a"), xb.vars("id")), xb.index(xb.vars("b"), xb.vars("id")) }));

    const program = kernel.to_opencl_string(&.{ *usize, *usize, *usize });
    defer allocator.free(program);
    const expected_program = "__kernel void vadd(__global uint *a, __global uint *b, __global uint *c) {uint id = get_global_id(0); c[id] = (a[id] + b[id]);}";

    try std.testing.expect(std.mem.eql(u8, program, expected_program));
}
