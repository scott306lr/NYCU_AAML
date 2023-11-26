#include "models/ds_cnn_stream_fe/ds_cnn.h"
#include <stdio.h>
#include <string.h>
#include "menu.h"
#include "models/ds_cnn_stream_fe/ds_cnn_stream_fe.h"
#include "models/label/label0_board.h"
#include "models/label/label11_board.h"
#include "models/label/label1_board.h"
#include "models/label/label6_board.h"
#include "models/label/label8_board.h"
#include "models/my_cycles.h"
#include "tflite.h"

// Initialize everything once
// deallocate tensors when done
static void ds_cnn_stream_fe_init(void) {
    tflite_load_model(ds_cnn_stream_fe, ds_cnn_stream_fe_len);
}

// Implement your design here
static void do_predict_fp_label(const float* label_data) {
    // set input
    tflite_set_input_float(label_data);

    // start classification
    reset_my_cycles();
    tflite_classify();

    // print cycles
    long long unsigned cycles = get_my_cycles();
    printf("MAC: %llu\n", cycles);

    // get output
    uint32_t output32[12];
    memcpy(output32, tflite_get_output_float(), 12 * sizeof(float));
    // memcpy(output32, tflite_get_output(), 12 * sizeof(int8_t));

    // print output
    // printf("    Results are: \n");
    for (int i = 0; i < 12; i++) {
        printf("%d : 0x%08lx, \n", i, output32[i]);
    }
}

static void do_predict_all_labels() {
    // printf("Label0: \n");
    // do_predict_fp_label(label0_data);

    // printf("Label1: \n");
    // do_predict_fp_label(label1_data);

    // printf("Label6: \n");
    // do_predict_fp_label(label6_data);

    do_predict_fp_label(label8_data);
    printf("---- Label8. \n");

    // printf("Label11: \n");
    // do_predict_fp_label(label11_data);
}

static struct Menu MENU = {
    "Tests for ds_cnn_stream_fe",
    "ds_cnn_stream_fe",
    {
        MENU_ITEM('1', "Predict all label data", do_predict_all_labels),
        MENU_END,
    },
};

// For integration into menu system
void ds_cnn_stream_fe_menu() {
    ds_cnn_stream_fe_init();
    menu_run(&MENU);
}
