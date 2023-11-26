module Cfu (
    input               cmd_valid,
    output              cmd_ready,
    input      [9:0]    cmd_payload_function_id,
    input      [31:0]   cmd_payload_inputs_0,
    input      [31:0]   cmd_payload_inputs_1,
    output reg          rsp_valid,
    input               rsp_ready,
    output reg [31:0]   rsp_payload_outputs_0,
    input               reset,
    input               clk
  );

  // SIMD multiply step:
  reg signed [8:0] input_offset;
  wire signed [15:0] prod_0, prod_1, prod_2, prod_3;
  assign prod_0 =  ($signed(cmd_payload_inputs_0[7 : 0]) + input_offset)
         * $signed(cmd_payload_inputs_1[7 : 0]);
  assign prod_1 =  ($signed(cmd_payload_inputs_0[15: 8]) + input_offset)
         * $signed(cmd_payload_inputs_1[15: 8]);
  assign prod_2 =  ($signed(cmd_payload_inputs_0[23:16]) + input_offset)
         * $signed(cmd_payload_inputs_1[23:16]);
  assign prod_3 =  ($signed(cmd_payload_inputs_0[31:24]) + input_offset)
         * $signed(cmd_payload_inputs_1[31:24]);

  wire signed [31:0] sum_prods;
  assign sum_prods = prod_0 + prod_1 + prod_2 + prod_3;

  // Only not ready for a command when we have a response.
  assign cmd_ready = ~rsp_valid;

  // Constants for Function IDs
  parameter FUNC_ID_ADD = 7'd0;
  parameter FUNC_ID_RESET = 7'd1;
  parameter FUNC_ID_SET_OFFSET = 7'd2;


  always @(posedge clk)
  begin
    if (reset)
    begin
      input_offset <= 9'd128;
      rsp_payload_outputs_0 <= 32'b0;
      rsp_valid <= 1'b0;
    end
    else if (rsp_valid)
    begin
      // Waiting to hand off response to CPU.
      rsp_valid <= ~rsp_ready;
    end
    else if (cmd_valid)
    begin
      rsp_valid <= 1'b1;
      //if func7 is 2, then set input_offset to the first input
      if (cmd_payload_function_id[9:3] == FUNC_ID_ADD)
      begin
        rsp_payload_outputs_0 <= rsp_payload_outputs_0 + sum_prods;
      end
      else if (cmd_payload_function_id[9:3] == FUNC_ID_RESET)
      begin
        rsp_payload_outputs_0 <= 32'b0;
      end
      else if (cmd_payload_function_id[9:3] == FUNC_ID_SET_OFFSET)
      begin
        input_offset <= cmd_payload_inputs_0[8:0];
      end
    end
  end
endmodule
