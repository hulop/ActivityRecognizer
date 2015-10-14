/*******************************************************************************
* Copyright (c) 2015 IBM Corporation
*
* Permission is hereby granted, free of charge, to any person obtaining a copy
* of this software and associated documentation files (the "Software"), to deal
* in the Software without restriction, including without limitation the rights
* to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
* copies of the Software, and to permit persons to whom the Software is
* furnished to do so, subject to the following conditions:
*
* The above copyright notice and this permission notice shall be included in
* all copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
* AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
* OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
* THE SOFTWARE.
*******************************************************************************/

#include "GTEAGazePlusUtils.h"

map<string, string> GTEAGazePlusUtils::getGteaGazeNormalizeVerbObjectMap() {
	map<string, string> normMap;

	// normalized verb and object pairs that are used in ECCV 2012 paper.

	// difficult
	{
		string take_cup_counter = "take(cup,counter)";
		normMap["take(cup,counter)"] = take_cup_counter;
		normMap["take(cup, counter)"] = take_cup_counter;
	}
	// difficult
	{
		string take_plate_counter = "take(plate,counter)";
		normMap["take(plate,counter)"] = take_plate_counter;
		normMap["take(plate, counter)"] = take_plate_counter;
	}
	// difficult
	{
		string take_bowl_counter = "take(bowl,counter)";
		normMap["take(bowl,counter)"] = take_bowl_counter;
		normMap["take(bowl, counter)"] = take_bowl_counter;
	}
	// difficult
	{
		string take_knife_counter = "take(knife,counter)";
		normMap["take(knife,counter)"] = take_knife_counter;
		normMap["take(knife, counter)"] = take_knife_counter;
	}
	{
		string take_bread_breadcontainer = "take(bread,bread container)";
		normMap["take(bread,bread container)"] = take_bread_breadcontainer;
		normMap["take(bread, bread container)"] = take_bread_breadcontainer;
		normMap["take(bread slices, bread container)"] = take_bread_breadcontainer;
	}
	// difficult
	{
		string take_peanutbuttercontainer_counter = "take(peanut butter container,counter)";
		normMap["take(peanut butter container,counter)"] = take_peanutbuttercontainer_counter;
		normMap["take(peanut butter container, counter)"] = take_peanutbuttercontainer_counter;
		normMap["take(peanut butter container)"] = take_peanutbuttercontainer_counter;
	}
	{
		string open_peanutbuttercontainer = "open(peanut butter container)";
		normMap["open(peanut butter container)"] = open_peanutbuttercontainer;
	}
	// difficult
	{
		string transfer_peanutbutter_knife = "transfer(peanut butter,spoon,cup,blunt knife)";
		normMap["transfer(peanut butter,spoon,cup,blunt knife)"] = transfer_peanutbutter_knife;
		normMap["transfer(peanut butter,peanut butter jar,plate,knife)"] = transfer_peanutbutter_knife;
		normMap["transfer(peanut butter,peanut butter container,cup,blunt knife)"] = transfer_peanutbutter_knife;
		normMap["transfer(peanut butter mixture,cup,bread,blunt knife)"] = transfer_peanutbutter_knife;
	}
	// difficult
	{
		string spread_peanutbuttermixture_spoon = "spread(peanut butter mixture,bread,spoon)";
		normMap["spread(peanut butter mixture,plate,bread,spoon)"] = spread_peanutbuttermixture_spoon;
		normMap["spread(peanut butter mixture,bread,spoon)"] = spread_peanutbuttermixture_spoon;
		normMap["spread(peanut butter mixture, bowl, bread, spoon)"] = spread_peanutbuttermixture_spoon;
	}
	// difficult
	{
		string spread_jelly_container_counter = "take(jelly container,counter)";
		normMap["take(jelly container,counter)"] = spread_jelly_container_counter;
		normMap["take(jelly container, counter)"] = spread_jelly_container_counter;
	}
	// difficult
	{
		string open_jelly_container = "open(jelly container)";
		normMap["open(jelly container)"] = open_jelly_container;
	}
	// difficult
	{
		string transfer_jelly_knife = "transfer(jelly,jelly container,cup,blunt knife)";
		normMap["transfer(jelly,jelly container,cup,blunt knife)"] = transfer_jelly_knife;
		normMap["transfer(jelly,jelly container,bowl,knife)"] = transfer_jelly_knife;
	}
	// difficult
	{
		string close_jelly_container = "close(jelly container)";
		normMap["close(jelly container)"] = close_jelly_container;
	}
	{
		string compress_sandwich = "compress(sandwich)";
		normMap["compress(sandwich)"] = compress_sandwich;
	}
	// difficult
	{
		string close_peanutbutter = "close(peanut butter container)";
		normMap["close(peanut butter container)"] = close_peanutbutter;
		normMap["close(peanut butter-container)"] = close_peanutbutter;
	}
	// difficult
	{
		string take_milkcontainer_counter = "take(milk container,counter)";
		normMap["take(milk container,counter)"] = take_milkcontainer_counter;
		normMap["take(milk container, counter)"] = take_milkcontainer_counter;
		normMap["take(milk container)"] = take_milkcontainer_counter;
	}
	{
		string open_milkcontainer = "open(milk container)";
		normMap["open(milk container)"] = open_milkcontainer;
	}
	{
		string pour_milk_bowl = "pour(milk,milk container,bowl)";
		normMap["pour(milk,milk container,bowl)"] = pour_milk_bowl;
		normMap["pour(milk,milk container,bowl,hands)"] = pour_milk_bowl;
		normMap["pour(milk,milk container,bowl,spoon)"] = pour_milk_bowl;
		normMap["pour(milk, milk container, bowl, hands)"] = pour_milk_bowl;
	}
	{
		string close_milkcontainer = "close(milk container)";
		normMap["close(milk container)"] = close_milkcontainer;
	}
	{
		string take_turkey_turkeycontainer = "take(turkey,turkey container)";
		normMap["take(turkey,turkey container)"] = take_turkey_turkeycontainer;
		normMap["take(turkey, turkey container)"] = take_turkey_turkeycontainer;
		normMap["take(turkey slice, turkey container)"] = take_turkey_turkeycontainer;
	}
	// difficult
	{
		string close_turkeycontainer = "close(turkey container)";
		normMap["close(turkey container)"] = close_turkeycontainer;
	}
	{
		string take_cheesepacket_cheesecontainer = "take(cheese packet,cheese container)";
		normMap["take(cheese packet,cheese container)"] = take_cheesepacket_cheesecontainer;
		normMap["take(cheese packet, cheese container)"] = take_cheesepacket_cheesecontainer;
		normMap["take(cheese packet wrapper, cheese)"] = take_cheesepacket_cheesecontainer;
	}
	// difficult
	{
		string open_cheesepacket = "open(cheese packet)";
		normMap["open(cheese packet)"] = open_cheesepacket;
	}
	{
		string take_cheese_cheesepacket = "take(cheese,cheese packet)";
		normMap["take(cheese,cheese packet)"] = take_cheese_cheesepacket;
		normMap["take(cheese slice,cheese container)"] = take_cheese_cheesepacket;
		normMap["take(cheese slice, cheese packet)"] = take_cheese_cheesepacket;
	}

	// other normalized verb and object pairs
	/*
	{
		string move_around_pasta_spoon = "move around(pasta, pot, plastic spoon)";
		normMap["move around(pasta, pot, plastic spoon)"] = move_around_pasta_spoon;
		normMap["move around(pasta, pot, plastic holed spoon)"] = move_around_pasta_spoon;
	}
	{
		string close_oilcontainer = "close(oil container)";
		normMap["close(oil container)"] = close_oilcontainer;
	}
	{
		string open_oilcontainer = "open(oil container)";
		normMap["open(oil container)"] = open_oilcontainer;
	}
	{
		string cut_pepper_knife = "cut(pepper,knife)";
		normMap["cut(pepper,knife)"] = cut_pepper_knife;
		normMap["cut(pepper, knife)"] = cut_pepper_knife;
		normMap["cut(pepper, blunt knife)"] = cut_pepper_knife;
		normMap["cut(pepper slice,knife)"] = cut_pepper_knife;
		normMap["cut(pepper slice, knife)"] = cut_pepper_knife;
		normMap["cut(pepper piece, knife)"] = cut_pepper_knife;
	}
	{
		string move_around_patty_spoon = "move around(patty, skillet, plastic holed spoon)";
		normMap["move around(patty, skillet, plastic holed spoon)"] = move_around_patty_spoon;		
	}
	{
		string cut_lettuce_knife = "cut(lettuce,knife)";
		normMap["cut(lettuce,knife)"] = cut_lettuce_knife;
		normMap["cut(lettuce, knife)"] = cut_lettuce_knife;
		normMap["cut(lettuce,blunt knife)"] = cut_lettuce_knife;
	}
	{
		string flip_bacon_skillet = "flip(bacon,skillet,spatula)";
		normMap["flip(bacon,skillet,spatula)"] = flip_bacon_skillet;
		normMap["flip(bacon, skillet, spatula)"] = flip_bacon_skillet;
	}
	{
		string move_around_mushroom_hands = "move around(mushroom,skillet,hands)";
		normMap["move around(mushroom,skillet,hands)"] = move_around_mushroom_hands;
	}
	{
		string move_around_mushroom_spatula = "move around(mushroom,skillet,spatula)";
		normMap["move around(mushroom,skillet,spatula)"] = move_around_mushroom_spatula;
	}
	*/

	return normMap;
}