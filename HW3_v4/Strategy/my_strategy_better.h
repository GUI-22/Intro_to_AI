#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include "Judge.h"
#include "Point.h"
using namespace std;

int limit = 2.0 * CLOCKS_PER_SEC;
int max_search_time = 1e6;
double param = 0.7;
#define MACHINE_CHESS 2
#define USER_CHESS 1
#define BLANK_CHESS 0
double MINUS_INF = -10000000;


class Node {
    private:
        int w, h;
        //set_x和set_y为“刚刚走”的子，已经呈现在棋盘上
        int set_x, set_y;
        int search_time = 0;
        //判断该节点是否已经可以分出胜负
        int win = -5;
        //turn为刚刚走的set_x和set_y是谁走的（1为machine；0为user）
        int turn;
        int unexpanded_num = 0;
        //优势积分
        int goodness = 0;
  
    public:
        const int& get_w() {return w;}
        const int& get_h() {return h;}
        const int& get_set_x() {return set_x;}
        const int& get_set_y() {return set_y;}
        const int& get_turn() {return turn;}
        int& get_unexpanded_num() {return unexpanded_num;}
        const int& get_win() {return win;}
        int& get_goodness() {return goodness;}
        int& get_search_time() {return search_time;}

        int** board;
        int* top;
        Node* parent;
        Node** children;
        int* unexpanded_children;
        int is_urge;

    Node(int w, int h, int** board, int* top, int set_x, int set_y, int turn, Node* parent) {
        this->w = w;
        this->h = h;
        this->set_x = set_x;
        this->set_y = set_y;
        this->turn = turn;
        this->parent = parent;
        // 每个节点的board top children 都应该地址不同
        this->board = new int*[h];
        for (int i = 0; i < h; i++) {
            this->board[i] = new int[w];
            for (int j = 0; j < w; j++) {
                this->board[i][j] = board[i][j];
            } 
        }
        this->top = new int[w];
        for (int i = 0; i < w; i++) {
            this->top[i] = top[i];
        }
        this->children = new Node*[w];
        for(int i = 0; i < w; i++) {
            this->children[i] = nullptr;
        }
        this->unexpanded_children = new int[w];
        for(int i = 0; i < w; i++) {
            if(top[i] > 0) {
                this->unexpanded_children[this->unexpanded_num] = i;
                this->unexpanded_num += 1;
            }
        }
        if (turn == 1 && machineWin(set_x, set_y, h, w, board)) {
            win = 1;
        }
        else if (isTie(w, top)) {
            win = 0;
        }
        else if (turn == 0 && userWin(set_x, set_y, h, w, board)) {
            win = -1;
        }
        this->is_urge = this->urge();
    }

    ~Node() {
        for (int i = 0; i < h; i++) {
            delete[] board[i];
        }
        for (int i = 0; i < w; i++) {
            if (children[i] != nullptr){
                delete children[i];
            }
        }
        delete[] top;
        delete[] board;
        delete[] children;
        delete[] unexpanded_children;
    }
    inline int win_in_one_step_machine() {
        for (int y = 0; y < this->get_w(); y++) {
            if (top[y] > 0) {
                int x = top[y] - 1;
                board[x][y] = MACHINE_CHESS;
                if (machineWin(x, y, this->get_h(), this->get_w(), board)) {
                    board[x][y] = BLANK_CHESS;
                    return y;
                }
                board[x][y] = BLANK_CHESS;
            }
        }
        return -1;
    }
    inline int win_in_one_step_user() {
        for (int y = 0; y < this->get_w(); y++) {
            if (top[y] > 0) {
                int x = top[y] - 1;
                board[x][y] = USER_CHESS;
                if (userWin(x, y, this->get_h(), this->get_w(), board)) {
                    board[x][y] = BLANK_CHESS;
                    return y;
                }
                board[x][y] = BLANK_CHESS;
            }
        }
        return -1;
    }
    int urge() {
        // turn==1 表示为自己（机器）走子
        //先判断我方是否可以立刻获胜，可以则走子
        //我方无一步必胜策略，则要防御对方的“一步必胜”
        int check;
        if (this->get_turn()==0){
            check = win_in_one_step_machine();
            if (check >= 0) {
                return check;
            }
            check = win_in_one_step_user();
            if (check >= 0) {
                return check;
            }
        } 
        else {
            check = win_in_one_step_user();
            if (check >= 0) {
                return check;
            }
            check = win_in_one_step_machine();
            if (check >= 0) {
                return check;
            }
        }
        return -1;
    }
 };

class UCT {
    private:
        int w, h;
        int skip_x, skip_y;
    public:
        long long total_nodes = 0;
        const int& get_w() {return w;}
        const int& get_h() {return h;}
        const int& get_skip_x() {return skip_x;}
        const int& get_skip_y() {return skip_y;}
        int* weight;


    UCT(int w, int h, int skip_x, int skip_y){
        this->w = w;
        this->h = h;
        this->skip_x = skip_x;
        this->skip_y = skip_y;
        this->weight = new int[w];
        this->weight[0] = 1;
        for (int i = 1; i < w; i++) {
            if (i < (w + 1) / 2) {
                this->weight[i] = i + 1;
            }
            else {
                this->weight[i] = this->weight[w - i - 1];
            }
        }
        for (int i = 1; i < w; i++) {
            this->weight[i] += this->weight[i - 1];
        }
    }

    ~UCT() {
        delete[] weight;
    }
    int search(Node* node) {
        int start = clock();
        while (clock() < start + limit) {
            Node* update_node = node;
            while(1) {
                if (update_node->get_win() != -5) break;
                if (update_node->get_unexpanded_num() > 0) {
                    update_node = expand(update_node);
                    break;
                }
                update_node = find_next_child_to_search(update_node);
            }
            
            int goodness = get_goodness(update_node);
            
            while(update_node != nullptr) {
                update_node->get_search_time() += 1;
                update_node->get_goodness() += goodness;
                update_node = update_node->parent;
            }
        }

        int temp = find_final_ans(node);
        return temp;
        
    }

    int find_final_ans(Node* curr_node) {
        Node* child = nullptr;
        int final_y = -1;
        double max_score = MINUS_INF;
        double score, part_one;
        for (int i = 0; i < curr_node->get_w(); i++) {
            if (curr_node->children[i] != nullptr) {
                part_one = (double)(curr_node->children[i]->get_goodness()) / (double)(curr_node->children[i]->get_search_time());
                score = (curr_node->get_turn() ? -1 : 1) * part_one;
                if (score > max_score) {
                    max_score = score;
                    child = curr_node->children[i];
                    final_y = curr_node->children[i]->get_set_y();
                }
            }
        }
        return final_y;
    }

    Node* find_next_child_to_search(Node* curr_node) {
        Node* child = nullptr;
        double max_score = MINUS_INF;
        double score, part_one, part_two;
        for (int i = 0; i < curr_node->get_w(); i++) {
            if (curr_node->children[i] != nullptr) {
                part_one = (double)(curr_node->children[i]->get_goodness()) / (double)(curr_node->children[i]->get_search_time());
                
                part_two = param * sqrt(2 * log((double)(curr_node->get_search_time())) / (double)(curr_node->children[i]->get_search_time()));
                
                score = (curr_node->get_turn() ? -1 : 1) * part_one + part_two;
                if (score > max_score) {
                    max_score = score;
                    child = curr_node->children[i];
                }
            }
        }
        return child;
    }
    
    //随机走子，以获得一个结果
    int get_goodness(Node* node) {
        int** board = new int*[h];
        for (int i = 0; i < h; i++) {
            board[i] = new int[w];
            for (int j = 0; j < w; j++) {
                board[i][j] = node->board[i][j];
            }
        }
        int* top = new int[w];
        for (int i = 0; i < w; i++) {
            top[i] = node->top[i];
        }
        int turn = node->get_turn();
        int w = node->get_w();
        int h = node->get_h();
        int goodness = node->get_win();
        
        while (goodness == -5)
        {
            int y = -1;
            while(true) {
                int rand_weight = rand() % this->weight[w - 1];
                for (int i = 0; i < node->get_w(); i++) {
                    if (this->weight[i] > rand_weight && top[i] > 0) {
                        y = i;
                        break;
                    }
                }
                if (y != -1) {
                    break;
                }
            }
            int x = top[y] - 1;
            //注意：走子之后更新board和top和turn，而更新top要考虑skip
            turn = 1 - turn;
            board[x][y] = turn + 1;
            top[y] -= 1;
            if (y == skip_y && top[y] - 1 == skip_x) {
                top[y] -= 1;
            }
            if (turn == 1 && machineWin(x, y, h, w, board)) {
                goodness = 1;
                break;
            }
            else if (turn == 0 && userWin(x, y, h, w, board)) {
                goodness = -1;
                break;
            }
            else if (isTie(w, top)) {
                goodness = 0;
                break;
            }
        }
        for (int i = 0; i < h; i++) {
            delete[] board[i];
        }
        delete[] board;
        delete[] top;
        return goodness;
    }

    //扩展节点
    Node* expand(Node* curr_node) {
        total_nodes += 1;
        int w = curr_node->get_w();
        int expand_x, expand_y, bias_top = 1;


        int rand_num = rand() % curr_node->get_unexpanded_num();
        expand_y = curr_node->unexpanded_children[rand_num];
        curr_node->get_unexpanded_num() -= 1;
        std::swap(curr_node->unexpanded_children[rand_num], curr_node->unexpanded_children[curr_node->get_unexpanded_num()]);
        expand_x = curr_node->top[expand_y] - 1;
        
        if (curr_node->top[expand_y] - 1 == this->get_skip_x() && expand_y == this->get_skip_y()) {
            bias_top = 2;
        }

        int** new_board = new int*[h];
        for (int i = 0; i < h; i++) {
            new_board[i] = new int[w];
            for (int j = 0; j < w; j++) {
                new_board[i][j] = curr_node->board[i][j];
            } 
        }
        int* new_top = new int[w];
        for (int i = 0; i < w; i++) {
            new_top[i] = curr_node->top[i];
        }

        new_top[expand_y] -= bias_top;
        new_board[expand_x][expand_y] = 2 - curr_node->get_turn();
        curr_node->children[expand_y] = new Node(w, curr_node->get_h(), new_board, new_top, expand_x, expand_y, 1 - curr_node->get_turn(), curr_node);

        delete[] new_top;
        for (int i = 0; i < h; i++) {
            delete[] new_board[i];
        }
        delete[] new_board;
        return curr_node->children[expand_y];
    }
};


