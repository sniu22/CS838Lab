
import java.util.ArrayList;
import java.util.Collections;

public class Cnn {

        public static void train_cph(Dataset trainset, Dataset tuneset, Dataset testset) {
            // define the structure of cnn
            Conv conv_1 = new Conv(3, 32, 16, 3, 0.1, 0.001, 16);
            Pool pool_1 = new Pool(2, 16, 32);
            Hidden hidden_1 = new Hidden(4096, 100, 0.5, 0.1, 0.001, 16);
            Out_Layer out_layer = new Out_Layer(100, "sigmoid", 0.001, 16);


            ArrayList<Instance> train = trainset.getImages();
            ArrayList<Instance> tune = tuneset.getImages();
            ArrayList<Instance> test = testset.getImages();


            for (int epoch = 0; epoch < 100; epoch++) {
                Collections.shuffle(train);
                int count = 0;
                hidden_1.renew_dropout(hidden_1.dropoutsize);

                int[][] res_mat = new int[6][6];

                for (int i = 0; i < train.size(); i++) {
                    double[][][] input = Tool.get_image2(train.get(i));
                    int label = Lab3.get_label(train.get(i).getLabel());
                    // forward
                    double[][][] c1_out = conv_1.compute(input);
                    double[][][] p1_out = pool_1.pool(c1_out);
                    double[] p_line = Tool.to_line(p1_out);
                    double[] h1_out = hidden_1.compute(p_line);
                    double[] f_out = out_layer.output(h1_out);


                    int prediction = Lab3.get_prediction(f_out);
                    count += (Lab3.get_prediction(f_out) == label) ? 1 : 0;
                    res_mat[label][prediction] += 1;

                    // Used for backward
                    double[] o_p1 = out_layer.f_back(f_out, h1_out, label);
                    double[] h_p1 = hidden_1.h_back(o_p1, h1_out, p_line);
                    double[][][] p1_p1 = pool_1.hp_back(h_p1, c1_out, p1_out);
                    //backward
                    out_layer.update(f_out, h1_out, label);
                    hidden_1.update(o_p1, h1_out, p_line);
                    conv_1.update(p1_p1, input, c1_out);
                }
                System.out.println("\nEpoch: " + epoch + " train: " + (double) count / train.size());

                for (int i = 0; i < res_mat.length; i++) {
                    for (int j = 0; j < res_mat[0].length; j++) {
                        System.out.print("\t" + res_mat[i][j]);
                    }
                    System.out.println();
                }

                hidden_1.renew_dropout(0);
                int[][] test_mat = new int[6][6];
                int test_count = 0;
                for (int i = 0; i < test.size(); i++) {
                    double[][][] input = Tool.get_image2(test.get(i));
                    int label = Lab3.get_label(test.get(i).getLabel());
                    // forward
                    double[][][] c1_out = conv_1.compute(input);
                    double[][][] p1_out = pool_1.pool(c1_out);
                    double[] p_line = Tool.to_line(p1_out);
                    double[] h1_out = hidden_1.compute(p_line);
                    double[] f_out = out_layer.output(h1_out);
                    int prediction = Lab3.get_prediction(f_out);
                    test_count += (Lab3.get_prediction(f_out) == label) ? 1 : 0;
                    test_mat[label][prediction] += 1;
                }

                System.out.println("\nEpoch: " + epoch + " test: " + (double) test_count / test.size());

                for (int i = 0; i < test_mat.length; i++) {
                    for (int j = 0; j < test_mat[0].length; j++) {
                        System.out.print("\t" + test_mat[i][j]);
                    }
                    System.out.println();
                }


            }

        }

        public static void train_h(Dataset trainset, Dataset tuneset, Dataset testset){
            Hidden hidden_1 = new Hidden(32*32*4, 100, 0.0, 0.1, 0.001, 16);
            Out_Layer out_layer = new Out_Layer(100, "softmax", 0.001, 16);


            ArrayList<Instance> train = (ArrayList<Instance>) trainset.getImages().clone();
            ArrayList<Instance> tune = (ArrayList<Instance>) tuneset.getImages().clone();
            ArrayList<Instance> test = (ArrayList<Instance>) testset.getImages().clone();

            for (int epoch = 0; epoch < 100; epoch++) {
                Collections.shuffle(train);
                int count = 0;
                hidden_1.renew_dropout(hidden_1.dropoutsize);
                int[][] res_mat = new int[6][6];
                for (int i = 0; i < train.size(); i++) {
                    double[][][] input = Tool.get_image2(train.get(i));
                    int label = Lab3.get_label(train.get(i).getLabel());
                    // forward

                    double[] p_line = Tool.to_line(input);

                    double[] h1_out = hidden_1.compute(p_line);
                    double[] f_out = out_layer.output(h1_out);

                    int prediction = Lab3.get_prediction(f_out);
                    count += (Lab3.get_prediction(f_out) == label) ? 1 : 0;
                    res_mat[label][prediction] += 1;

                    // Used for backward
                    double[] o_p1 = out_layer.f_back(f_out, h1_out, label);

                    //backward
                    out_layer.update(f_out, h1_out, label);
                    hidden_1.update(o_p1, h1_out, p_line);
                }
                System.out.println("\nEpoch: " + epoch + " , train: " + (double) count / train.size());

                for (int i = 0; i < res_mat.length; i++) {
                    for (int j = 0; j < res_mat[0].length; j++) {
                        System.out.print("\t" + res_mat[i][j]);
                    }
                    System.out.println();
                }
            }
        }
}


