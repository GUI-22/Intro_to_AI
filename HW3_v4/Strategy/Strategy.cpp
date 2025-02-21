#include <iostream>
#include <unistd.h>
#include "Point.h"
#include "Strategy.h"
#include "my_strategy_better.h"

using namespace std;

extern "C" Point *getPoint(const int M, const int N, const int *top, const int *_board,
						   const int lastX, const int lastY, const int noX, const int noY)
{
	/*
		不要更改这段代码
	*/
	int x = -1, y = -1; //最终将你的落子点存到x,y中
	int **board = new int *[M];
	for (int i = 0; i < M; i++)
	{
		board[i] = new int[N];
		for (int j = 0; j < N; j++)
		{
			board[i][j] = _board[i * N + j];
		}
	}

	/*
		根据你自己的策略来返回落子点,也就是根据你的策略完成对x,y的赋值
		该部分对参数使用没有限制，为了方便实现，你可以定义自己新的类、.h文件、.cpp文件
	*/
	//Add your own code below
	
   	int *top_copy = new int[N];
	for (int i = 0; i < N; i++)
	{
		top_copy[i] = top[i];
	}
	
	if (lastX == -1) {
		y = N / 2;
		x = top[y] - 1;
	}
	else {
		Node* node = new Node(N, M, board, top_copy, lastX, lastY, 0, nullptr);
		int urge;
		urge = node->urge();
		if (urge != -1) {
			y = urge;
			x = top[y] - 1;
		}
		else {
			UCT* uct_tree = new UCT(N, M, noX, noY);
			y = uct_tree->search(node);
			x = top[y] - 1;
			delete uct_tree;
		}
		delete node;
	}
   	delete[] top_copy;
	clearArray(M, N, board);
	return new Point(x, y);
}

/*
	getPoint函数返回的Point指针是在本so模块中声明的，为避免产生堆错误，应在外部调用本so中的
	函数来释放空间，而不应该在外部直接delete
*/
extern "C" void clearPoint(Point *p)
{
	delete p;
	return;
}

/*
	清除top和board数组
*/
void clearArray(int M, int N, int **board)
{
	for (int i = 0; i < M; i++)
	{
		delete[] board[i];
	}
	delete[] board;
}

/*
	添加你自己的辅助函数，你可以声明自己的类、函数，添加新的.h .cpp文件来辅助实现你的想法
*/
