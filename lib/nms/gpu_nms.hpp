#if defined(__linux__)
  #define NMS_TYPE int
#endif
#if defined(_WIN64)
  #define NMS_TYPE long
#endif

void _nms(NMS_TYPE* keep_out, int* num_out, const float* boxes_host, int boxes_num,
          int boxes_dim, float nms_overlap_thresh, int device_id);
