import  {NextResponse} from "next/server";
import {generateModelText} from "../../lib/openFinetunedAPIai";

export async function POST(req: Request){
    const body = await req.json();
    const model_text = body.model_text;
    const img_path = body.img_path;
    console.log(model_text);
    // trigger the generateModelText fxn
    const modelExplaination = await generateModelText(model_text, img_path);
    if(!modelExplaination) {
        return new NextResponse('failed to generate model_explainations', {status: 500});
    }

    return new NextResponse(modelExplaination);
}