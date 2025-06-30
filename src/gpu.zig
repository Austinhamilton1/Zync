const std = @import("std");

const ASTNode = struct {
    // The type of node this is.
    const AST_node_type = enum {
        var_leaf,
        const_leaf,
        assign,
        binary_op,
        compare_op,
        branch,
        while_loop,
        for_loop,
    };

    const BinaryOp = enum { add, sub, mul, div, rem };
    const CompareOp = enum { lt, gt, lte, gte, eq, neq };

    node_type: AST_node_type,
    variable: *const [1:0]u8 = undefined,
    constant: usize = undefined,
    assign_op: struct {
        lhs: *ASTNode,
        rhs: *ASTNode,
    } = undefined,
    binary_op: struct {
        op: BinaryOp,
        lhs: *ASTNode,
        rhs: *ASTNode,
    } = undefined,
    compare_op: struct {
        op: CompareOp,
        lhs: *ASTNode,
        rhs: *ASTNode,
    } = undefined,
    branch: struct {
        condition: *ASTNode,
        if_body: *ASTNode,
        else_body: ?*ASTNode,
    } = undefined,
    while_loop: struct {
        condition: *ASTNode,
        body: *ASTNode,
    } = undefined,
    for_loop: struct {
        condition: *ASTNode,
        body: *ASTNode,
    } = undefined,

    /// Convert an ASTNode to a string. This string method creates an OpenCL C string.
    /// Arguments:
    ///     allocator: std.mem.Allocator - An allocator to create the string with.
    /// Returns:
    ///     ![]const u8 - The string representation of the ASTNode.
    pub fn to_string(self: @This(), allocator: std.mem.Allocator) ![]const u8 {
        switch (self.node_type) {
            .var_leaf => {
                return try std.fmt.allocPrint(allocator, "{any}", .{self.variable});
            },
            .const_leaf => {
                return try std.fmt.allocPrint(allocator, "{any}", .{self.constant});
            },
            else => {
                return "";
            },
            // .assign => {
            //     return try std.fmt.allocPrint(allocator, "{any} = {any};", .{ self.assign_op.lhs.to_string(allocator), self.assign_op.rhs.to_string(allocator) });
            // },
            // .binary_op => {
            //     const op = switch (self.binary_op.op) {
            //         .add => "+",
            //         .sub => "-",
            //         .mul => "*",
            //         .div => "/",
            //         .rem => "%",
            //     };

            //     return try std.fmt.allocPrint(allocator, "{any} {any} {any}", .{ self.binary_op.lhs.to_string(allocator), op, self.binary_op.rhs.to_string(allocator) });
            // },
            // .compare_op => {
            //     const op = switch (self.compare_op.op) {
            //         .lt => "<",
            //         .gt => ">",
            //         .lte => "<=",
            //         .gte => ">=",
            //         .eq => "==",
            //         .neq => "!=",
            //     };

            //     return try std.fmt.allocPrint(allocator, "{any} {any} {any}", .{ self.compare_op.lhs.to_string(allocator), op, self.compare_op.rhs.to_string(allocator) });
            // },
            // .branch => {
            //     if (self.branch.else_body) |else_body| {
            //         return try std.fmt.allocPrint(allocator, "if({any}){{{any}}}else{{{any}}}", .{ self.branch.condition.to_string(allocator), self.branch.if_body.to_string(allocator), else_body.to_string(allocator) });
            //     } else {
            //         return try std.fmt.allocPrint(allocator, "if({any}){{{any}}}", .{ self.branch.condition.to_string(allocator), self.branch.if_body.to_string(allocator) });
            //     }
            // },
            // .while_loop => {
            //     return try std.fmt.allocPrint(allocator, "while({any}){{{any}}}", .{ self.while_loop.condition.to_string(allocator), self.while_loop.body.to_string(allocator) });
            // },
            // .for_loop => {
            //     return try std.fmt.allocPrint(allocator, "for({any}){{{any}}}", .{ self.for_loop.condition.to_string(allocator), self.for_loop.body.to_string(allocator) });
            // },
        }
    }
};

test "ASTNode test" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();

    const node1 = try allocator.create(ASTNode);
    const node2 = try allocator.create(ASTNode);
    const node3 = try allocator.create(ASTNode);
    const node4 = try allocator.create(ASTNode);
    const node5 = try allocator.create(ASTNode);
    const node6 = try allocator.create(ASTNode);
    const node7 = try allocator.create(ASTNode);
    defer allocator.free(node1);
    defer allocator.free(node2);
    defer allocator.free(node3);
    defer allocator.free(node4);
    defer allocator.free(node5);
    defer allocator.free(node6);
    defer allocator.free(node7);

    // node1.* = .{
    //     .node_type = .const_leaf,
    //     .constant = 10,
    // };

    // std.debug.print("{s}", .{try node1.to_string(allocator)});

    // node2.* = .{
    //     .node_type = .const_leaf,
    //     .constant = 0,
    // };

    // node3.* = .{
    //     .node_type = .const_leaf,
    //     .constant = 1,
    // };

    // node4.* = .{
    //     .node_type = .var_leaf,
    //     .variable = "i",
    // };

    // node5.* = .{
    //     .node_type = .compare_op,
    //     .compare_op = .{
    //         .op = .lt,
    //         .lhs = node4,
    //         .rhs = node1,
    //     },
    // };

    // node6.* = .{
    //     .node_type = .binary_op,
    //     .binary_op = .{
    //         .op = .add,
    //         .lhs = node4,
    //         .rhs = node3,
    //     },
    // };

    // node7.* = .{
    //     .node_type = .assign,
    //     .assign_op = .{
    //         .lhs = node4,
    //         .rhs = node6,
    //     },
    // };

    // node7.* = .{
    //     .node_type = .while_loop,
    //     .while_loop = .{
    //         .condition = node5,
    //         .body = node7,
    //     },
    // };

    // std.debug.print("{s}\n", .{node7.to_string(allocator)});
}
